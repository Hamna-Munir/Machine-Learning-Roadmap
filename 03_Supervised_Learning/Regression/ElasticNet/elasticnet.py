# =============================================================================
# 📦 ElasticNet Regression — Reusable ML Script
# =============================================================================
# Author   : Hamna Munir
# Topic    : 03_Supervised_Learning / Regression / ElasticNet
# File     : elasticnet.py
# =============================================================================

# -----------------------------------------------------------------------------
# 📚 Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.linear_model import (
    ElasticNet, ElasticNetCV,
    Lasso, Ridge, LinearRegression
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    KFold, GridSearchCV
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    r2_score, mean_absolute_percentage_error
)
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 🔧 1. TRAIN ELASTICNET (Fixed Hyperparameters)
# =============================================================================

def train_elasticnet(X_train: pd.DataFrame,
                      X_test: pd.DataFrame,
                      y_train: pd.Series,
                      y_test: pd.Series,
                      alpha: float = 0.1,
                      l1_ratio: float = 0.5,
                      max_iter: int = 10000,
                      scale: bool = True) -> dict:
    """
    Trains an ElasticNet Regression model with fixed alpha and l1_ratio.

    Formula:
        Loss = RSS + alpha × [l1_ratio × Σ|βⱼ| + (1−l1_ratio) × Σβⱼ²]

    Key properties:
        l1_ratio = 1.0  →  pure Lasso  (maximum sparsity)
        l1_ratio = 0.0  →  pure Ridge  (no sparsity)
        l1_ratio = 0.5  →  equal mix   (sklearn default)

    Best for:
        - Datasets with both correlated AND irrelevant features
        - High-dimensional data (p >> n)
        - When Lasso is unstable due to correlated features

    Args:
        X_train   : Training features DataFrame
        X_test    : Test features DataFrame
        y_train   : Training target Series
        y_test    : Test target Series
        alpha     : Overall regularization strength (default: 0.1)
        l1_ratio  : L1 vs L2 mix — 0=Ridge, 1=Lasso (default: 0.5)
        max_iter  : Maximum iterations for convergence (default: 10000)
        scale     : Whether to StandardScale features (default: True)

    Returns:
        Dictionary with model, predictions, coefficients, and metrics
    """
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", ElasticNet(
        alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter
    )))

    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)

    model        = pipe.named_steps["model"]
    y_pred_train = pipe.predict(X_train)
    y_pred_test  = pipe.predict(X_test)
    metrics      = _evaluate(y_train, y_pred_train, y_test, y_pred_test)
    coefs        = pd.Series(model.coef_, index=X_train.columns)
    zero_feats   = coefs[coefs == 0].index.tolist()
    active_feats = coefs[coefs != 0].index.tolist()

    print(f"[ElasticNet] alpha={alpha} | l1_ratio={l1_ratio}")
    print(f"  Active features : {len(active_feats)}  → {active_feats}")
    print(f"  Zeroed features : {len(zero_feats)}")
    print(f"  Sparsity        : {len(zero_feats)/len(X_train.columns)*100:.1f}%")
    _print_metrics(metrics)

    return {
        "pipeline"       : pipe,
        "model"          : model,
        "y_pred_train"   : y_pred_train,
        "y_pred_test"    : y_pred_test,
        "metrics"        : metrics,
        "coefficients"   : coefs,
        "active_features": active_feats,
        "zero_features"  : zero_feats,
        "sparsity_pct"   : round(len(zero_feats) / len(X_train.columns) * 100, 2),
        "alpha"          : alpha,
        "l1_ratio"       : l1_ratio,
    }


# =============================================================================
# 🔧 2. ELASTICNETCV — AUTO HYPERPARAMETER SELECTION
# =============================================================================

def train_elasticnet_cv(X_train: pd.DataFrame,
                         X_test: pd.DataFrame,
                         y_train: pd.Series,
                         y_test: pd.Series,
                         l1_ratios: list = None,
                         alphas: list = None,
                         cv: int = 5,
                         max_iter: int = 10000) -> dict:
    """
    Trains ElasticNet using built-in cross-validation to select
    the best alpha AND l1_ratio simultaneously.

    ElasticNetCV uses warm-start coordinate descent for efficiency —
    faster than GridSearchCV for this specific model.

    Args:
        X_train   : Training features DataFrame
        X_test    : Test features DataFrame
        y_train   : Training target Series
        y_test    : Test target Series
        l1_ratios : L1 ratio values to try (default: [0.1,0.5,0.7,0.9,0.95,1.0])
        alphas    : Alpha values (default: auto log-scale)
        cv        : Number of CV folds (default: 5)
        max_iter  : Max iterations (default: 10000)

    Returns:
        Dictionary with best alpha, best l1_ratio, model, and metrics
    """
    if l1_ratios is None:
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    model = ElasticNetCV(
        l1_ratio=l1_ratios,
        alphas=alphas,
        cv=cv,
        max_iter=max_iter,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_tr_sc, y_train)

    best_alpha    = model.alpha_
    best_l1_ratio = model.l1_ratio_
    y_pred_train  = model.predict(X_tr_sc)
    y_pred_test   = model.predict(X_te_sc)
    metrics       = _evaluate(y_train, y_pred_train, y_test, y_pred_test)
    coefs         = pd.Series(model.coef_, index=X_train.columns)
    zero_feats    = coefs[coefs == 0].index.tolist()
    active_feats  = coefs[coefs != 0].index.tolist()

    print(f"[ElasticNetCV] Best alpha: {best_alpha:.6f} | "
          f"Best l1_ratio: {best_l1_ratio} | cv={cv}")
    print(f"  Active features : {len(active_feats)}")
    print(f"  Zeroed features : {len(zero_feats)}")
    _print_metrics(metrics)

    return {
        "model"          : model,
        "scaler"         : scaler,
        "best_alpha"     : best_alpha,
        "best_l1_ratio"  : best_l1_ratio,
        "y_pred_train"   : y_pred_train,
        "y_pred_test"    : y_pred_test,
        "metrics"        : metrics,
        "coefficients"   : coefs,
        "active_features": active_feats,
        "zero_features"  : zero_feats,
    }


# =============================================================================
# 🔧 3. GRIDSEARCHCV — JOINT ALPHA + L1_RATIO TUNING
# =============================================================================

def tune_elasticnet(X_train: pd.DataFrame,
                     y_train: pd.Series,
                     alphas: list = None,
                     l1_ratios: list = None,
                     cv: int = 5,
                     scoring: str = "r2") -> dict:
    """
    Tunes both alpha and l1_ratio using GridSearchCV with K-Fold CV.

    Args:
        X_train   : Training features DataFrame
        y_train   : Training target Series
        alphas    : Alpha values to search (default: log-scale)
        l1_ratios : L1 ratio values to search
        cv        : Number of CV folds (default: 5)
        scoring   : Scoring metric (default: 'r2')

    Returns:
        Dictionary with best parameters, score, and GridSearchCV object
    """
    if alphas is None:
        alphas = np.logspace(-3, 2, 15).tolist()
    if l1_ratios is None:
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  ElasticNet(max_iter=10000))
    ])

    param_grid = {
        "model__alpha"    : alphas,
        "model__l1_ratio" : l1_ratios,
    }

    grid = GridSearchCV(
        pipe, param_grid,
        cv=cv, scoring=scoring,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best_alpha    = grid.best_params_["model__alpha"]
    best_l1_ratio = grid.best_params_["model__l1_ratio"]
    best_score    = grid.best_score_

    print(f"[GridSearchCV ElasticNet] Best alpha: {best_alpha:.6f} | "
          f"Best l1_ratio: {best_l1_ratio} | CV {scoring}: {best_score:.4f}")

    return {
        "grid"          : grid,
        "best_alpha"    : best_alpha,
        "best_l1_ratio" : best_l1_ratio,
        "best_score"    : best_score,
        "best_pipeline" : grid.best_estimator_,
        "cv_results"    : pd.DataFrame(grid.cv_results_),
    }


# =============================================================================
# 🔧 4. L1_RATIO SENSITIVITY ANALYSIS
# =============================================================================

def l1_ratio_sensitivity(X_train: pd.DataFrame,
                          X_test: pd.DataFrame,
                          y_train: pd.Series,
                          y_test: pd.Series,
                          alpha: float = 0.1,
                          l1_ratios: list = None) -> pd.DataFrame:
    """
    Evaluates ElasticNet across a range of l1_ratio values while
    keeping alpha fixed — shows the Lasso↔Ridge interpolation.

    Args:
        X_train   : Training features DataFrame
        X_test    : Test features DataFrame
        y_train   : Training target Series
        y_test    : Test target Series
        alpha     : Fixed alpha value (default: 0.1)
        l1_ratios : L1 ratio values to evaluate

    Returns:
        DataFrame with metrics and sparsity per l1_ratio
    """
    if l1_ratios is None:
        l1_ratios = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    rows = []
    for r in l1_ratios:
        model = ElasticNet(alpha=alpha, l1_ratio=r, max_iter=10000)
        model.fit(X_tr_sc, y_train)

        y_tr_p = model.predict(X_tr_sc)
        y_te_p = model.predict(X_te_sc)
        coefs  = model.coef_
        n_zero = (coefs == 0).sum()

        rows.append({
            "l1_ratio"    : r,
            "Behavior"    : "→ Ridge" if r == 0.0 else ("→ Lasso" if r == 1.0 else "Mixed"),
            "Train R²"    : round(r2_score(y_train, y_tr_p), 4),
            "Test R²"     : round(r2_score(y_test,  y_te_p), 4),
            "RMSE"        : round(np.sqrt(mean_squared_error(y_test, y_te_p)), 4),
            "Active Feats": int(len(coefs) - n_zero),
            "Zero Feats"  : int(n_zero),
            "Sparsity %"  : round(n_zero / len(coefs) * 100, 1),
        })

    df = pd.DataFrame(rows)
    print(f"l1_ratio Sensitivity (alpha={alpha}):")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 5. ALPHA SENSITIVITY ANALYSIS
# =============================================================================

def alpha_sensitivity(X_train: pd.DataFrame,
                       X_test: pd.DataFrame,
                       y_train: pd.Series,
                       y_test: pd.Series,
                       l1_ratio: float = 0.5,
                       alphas: list = None) -> pd.DataFrame:
    """
    Evaluates ElasticNet across a range of alpha values while
    keeping l1_ratio fixed.

    Args:
        X_train   : Training features DataFrame
        X_test    : Test features DataFrame
        y_train   : Training target Series
        y_test    : Test target Series
        l1_ratio  : Fixed l1_ratio value (default: 0.5)
        alphas    : Alpha values to evaluate

    Returns:
        DataFrame with metrics and sparsity per alpha
    """
    if alphas is None:
        alphas = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    rows = []
    for a in alphas:
        if a == 0.0:
            model = LinearRegression()
        else:
            model = ElasticNet(alpha=a, l1_ratio=l1_ratio, max_iter=10000)
        model.fit(X_tr_sc, y_train)

        y_tr_p = model.predict(X_tr_sc)
        y_te_p = model.predict(X_te_sc)
        coefs  = model.coef_
        n_zero = (coefs == 0).sum()

        rows.append({
            "Alpha"       : a,
            "Train R²"    : round(r2_score(y_train, y_tr_p), 4),
            "Test R²"     : round(r2_score(y_test,  y_te_p), 4),
            "RMSE"        : round(np.sqrt(mean_squared_error(y_test, y_te_p)), 4),
            "Active Feats": int(len(coefs) - n_zero),
            "Zero Feats"  : int(n_zero),
            "Sparsity %"  : round(n_zero / len(coefs) * 100, 1),
            "L1 Norm"     : round(np.sum(np.abs(coefs)), 4),
        })

    df = pd.DataFrame(rows)
    print(f"Alpha Sensitivity (l1_ratio={l1_ratio}):")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 6. COEFFICIENT PATH
# =============================================================================

def coefficient_path(X_train: pd.DataFrame,
                      y_train: pd.Series,
                      l1_ratio: float = 0.5,
                      alphas: list = None) -> pd.DataFrame:
    """
    Computes the ElasticNet coefficient path — how each feature's
    coefficient changes as alpha increases.

    Args:
        X_train  : Training features DataFrame
        y_train  : Training target Series
        l1_ratio : Fixed l1_ratio for the path (default: 0.5)
        alphas   : Alpha values for the path (default: log-scale)

    Returns:
        DataFrame with coefficient values per feature per alpha
    """
    if alphas is None:
        alphas = np.logspace(-4, 2, 60).tolist()

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_train)

    path_dict = {}
    for a in alphas:
        model = ElasticNet(alpha=a, l1_ratio=l1_ratio, max_iter=20000)
        model.fit(X_sc, y_train)
        path_dict[a] = dict(zip(X_train.columns, model.coef_))

    path_df = pd.DataFrame(path_dict).T
    path_df.index.name = "Alpha"

    print(f"[ElasticNet Coef Path] l1_ratio={l1_ratio} | "
          f"{len(alphas)} alpha values")
    for feat in X_train.columns:
        for a, vals in path_dict.items():
            if vals[feat] == 0.0:
                print(f"  '{feat}' → zeroed at alpha ≈ {a:.6f}")
                break
        else:
            print(f"  '{feat}' → never zeroed in this range")

    return path_df


# =============================================================================
# 🔧 7. COMPARE LINEAR vs RIDGE vs LASSO vs ELASTICNET
# =============================================================================

def compare_all_models(X_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        y_train: pd.Series,
                        y_test: pd.Series,
                        ridge_alpha: float = 1.0,
                        lasso_alpha: float = 0.1,
                        en_alpha: float = 0.1,
                        en_l1_ratio: float = 0.5) -> pd.DataFrame:
    """
    Side-by-side comparison of all four regression variants:
    Linear, Ridge, Lasso, and ElasticNet.

    Args:
        X_train     : Training features DataFrame
        X_test      : Test features DataFrame
        y_train     : Training target Series
        y_test      : Test target Series
        ridge_alpha : Alpha for Ridge
        lasso_alpha : Alpha for Lasso
        en_alpha    : Alpha for ElasticNet
        en_l1_ratio : l1_ratio for ElasticNet

    Returns:
        DataFrame with comparison metrics for all four models
    """
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    configs = [
        ("Linear Regression",                     LinearRegression()),
        (f"Ridge        (α={ridge_alpha})",        Ridge(alpha=ridge_alpha)),
        (f"Lasso        (α={lasso_alpha})",        Lasso(alpha=lasso_alpha, max_iter=10000)),
        (f"ElasticNet   (α={en_alpha}, ρ={en_l1_ratio})",
         ElasticNet(alpha=en_alpha, l1_ratio=en_l1_ratio, max_iter=10000)),
    ]

    rows = []
    for name, model in configs:
        model.fit(X_tr_sc, y_train)
        y_tr  = model.predict(X_tr_sc)
        y_te  = model.predict(X_te_sc)
        coefs = model.coef_
        n_zero = int((coefs == 0).sum())

        rows.append({
            "Model"       : name,
            "Train R²"    : round(r2_score(y_train, y_tr), 4),
            "Test R²"     : round(r2_score(y_test,  y_te), 4),
            "RMSE"        : round(np.sqrt(mean_squared_error(y_test, y_te)), 4),
            "MAE"         : round(mean_absolute_error(y_test, y_te), 4),
            "Zero Coefs"  : n_zero,
            "Active Feats": int(len(coefs) - n_zero),
            "Sparsity %"  : round(n_zero / len(coefs) * 100, 1),
        })

    comp_df = pd.DataFrame(rows)
    print("Full Model Comparison — Linear vs Ridge vs Lasso vs ElasticNet:")
    print(comp_df.to_string(index=False))
    return comp_df


# =============================================================================
# 🔧 8. CROSS-VALIDATION
# =============================================================================

def cross_validate_elasticnet(X: pd.DataFrame,
                                y: pd.Series,
                                alpha: float = 0.1,
                                l1_ratio: float = 0.5,
                                cv: int = 5,
                                scoring: str = "r2") -> dict:
    """
    Performs K-Fold Cross-Validation on ElasticNet Pipeline.

    Args:
        X        : Full feature DataFrame
        y        : Full target Series
        alpha    : ElasticNet alpha (default: 0.1)
        l1_ratio : ElasticNet l1_ratio (default: 0.5)
        cv       : Number of CV folds (default: 5)
        scoring  : Scoring metric (default: 'r2')

    Returns:
        Dictionary with fold scores, mean, and std
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000))
    ])
    kf     = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=kf, scoring=scoring)

    result = {
        "scores"      : scores,
        "mean"        : scores.mean(),
        "std"         : scores.std(),
        "fold_results": {f"Fold {i+1}": round(s, 4) for i, s in enumerate(scores)},
    }

    print(f"[ElasticNet CV] alpha={alpha} | l1_ratio={l1_ratio} | K={cv}")
    print(f"  {scoring.upper()}: {scores.mean():.4f} ± {scores.std():.4f}")
    for fold, score in result["fold_results"].items():
        print(f"  {fold}: {score}")

    return result


# =============================================================================
# 🔧 9. EVALUATION METRICS
# =============================================================================

def evaluate_elasticnet(y_train, y_pred_train, y_test, y_pred_test) -> dict:
    """
    Computes regression evaluation metrics for train and test sets.
    Metrics: MAE, MSE, RMSE, MAPE, R²
    """
    def metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return {
            "MAE"  : round(mean_absolute_error(y_true, y_pred), 4),
            "MSE"  : round(mse, 4),
            "RMSE" : round(np.sqrt(mse), 4),
            "MAPE" : round(mean_absolute_percentage_error(y_true, y_pred)*100, 4),
            "R²"   : round(r2_score(y_true, y_pred), 4),
        }
    return {"train": metrics(y_train, y_pred_train),
            "test" : metrics(y_test,  y_pred_test)}


# =============================================================================
# 🔧 HELPERS
# =============================================================================

def _evaluate(y_train, y_pred_train, y_test, y_pred_test):
    return evaluate_elasticnet(y_train, y_pred_train, y_test, y_pred_test)


def _print_metrics(metrics: dict) -> None:
    for split in ["train", "test"]:
        m = metrics[split]
        print(f"  [{split.upper():5s}] MAE={m['MAE']:>10.4f} | "
              f"RMSE={m['RMSE']:>10.4f} | R²={m['R²']:>7.4f}")


# =============================================================================
# 🚀 MAIN — Demo with Synthetic Dataset
# =============================================================================

if __name__ == "__main__":

    # ── Synthetic dataset: correlated + irrelevant features ───────────────
    np.random.seed(42)
    n = 400

    base   = np.random.randn(n)
    X = pd.DataFrame({
        "Feat_A"  : base + np.random.randn(n) * 0.1,        # correlated group
        "Feat_A2" : base + np.random.randn(n) * 0.1,        # correlated with A
        "Feat_A3" : base + np.random.randn(n) * 0.15,       # correlated with A
        "Feat_B"  : np.random.randn(n),                      # independent relevant
        "Feat_C"  : np.random.randn(n),                      # independent relevant
        "Noise_1" : np.random.randn(n),                      # irrelevant
        "Noise_2" : np.random.randn(n),                      # irrelevant
        "Noise_3" : np.random.randn(n),                      # irrelevant
        "Noise_4" : np.random.randn(n),                      # irrelevant
        "Noise_5" : np.random.randn(n),                      # irrelevant
    })

    y = pd.Series(
        3.0 * X["Feat_A"]
        + 2.5 * X["Feat_B"]
        - 1.8 * X["Feat_C"]
        + np.random.randn(n) * 0.5,
        name="Target"
    )

    print("=" * 65)
    print("📊 Dataset Info — Correlated + Irrelevant Features")
    print("=" * 65)
    print(f"Shape    : {X.shape}")
    print(f"Relevant : Feat_A (group: A, A2, A3), Feat_B, Feat_C")
    print(f"Noise    : Noise_1 to Noise_5")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── 1. ElasticNet fixed ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("1️⃣  ElasticNet (alpha=0.1, l1_ratio=0.5)")
    print("=" * 65)
    result = train_elasticnet(X_train, X_test, y_train, y_test,
                               alpha=0.1, l1_ratio=0.5)

    # ── 2. ElasticNetCV ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("2️⃣  ElasticNetCV — Auto Hyperparameter Selection")
    print("=" * 65)
    cv_result = train_elasticnet_cv(X_train, X_test, y_train, y_test)

    # ── 3. GridSearchCV ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("3️⃣  GridSearchCV — Joint Alpha + l1_ratio Tuning")
    print("=" * 65)
    gs_result = tune_elasticnet(X_train, y_train)

    # ── 4. l1_ratio Sensitivity ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("4️⃣  l1_ratio Sensitivity (alpha=0.1)")
    print("=" * 65)
    l1_df = l1_ratio_sensitivity(X_train, X_test, y_train, y_test, alpha=0.1)

    # ── 5. Alpha Sensitivity ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("5️⃣  Alpha Sensitivity (l1_ratio=0.5)")
    print("=" * 65)
    a_df = alpha_sensitivity(X_train, X_test, y_train, y_test, l1_ratio=0.5)

    # ── 6. Coefficient Path ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("6️⃣  Coefficient Regularization Path")
    print("=" * 65)
    path_df = coefficient_path(X_train, y_train, l1_ratio=0.5)

    # ── 7. Full Model Comparison ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("7️⃣  Full Model Comparison")
    print("=" * 65)
    compare_all_models(
        X_train, X_test, y_train, y_test,
        ridge_alpha=1.0,
        lasso_alpha=0.1,
        en_alpha=cv_result["best_alpha"],
        en_l1_ratio=cv_result["best_l1_ratio"]
    )

    # ── 8. Cross-Validation ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("8️⃣  5-Fold Cross-Validation")
    print("=" * 65)
    cross_validate_elasticnet(
        X, y,
        alpha=cv_result["best_alpha"],
        l1_ratio=cv_result["best_l1_ratio"],
        cv=5
    )

    print("\n✅ All ElasticNet techniques demonstrated successfully!")
