# =============================================================================
# 📦 Lasso Regression — Reusable ML Script
# =============================================================================
# Author   : Hamna Munir
# Topic    : 03_Supervised_Learning / Regression / Lasso_Regression
# File     : lasso_regression.py
# =============================================================================

# -----------------------------------------------------------------------------
# 📚 Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge
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
# 🔧 1. TRAIN LASSO REGRESSION (Fixed Alpha)
# =============================================================================

def train_lasso(X_train: pd.DataFrame,
                 X_test: pd.DataFrame,
                 y_train: pd.Series,
                 y_test: pd.Series,
                 alpha: float = 0.1,
                 max_iter: int = 10000,
                 scale: bool = True) -> dict:
    """
    Trains a Lasso Regression model with a fixed regularization alpha.

    Formula:
        Loss = RSS + alpha × Σ|βⱼ|

    Key property:
        Lasso zeros out irrelevant features automatically.
        Check result['zero_features'] to see which were eliminated.

    Best for:
        - High-dimensional data with many irrelevant features
        - Automatic feature selection
        - Sparse model solutions

    Args:
        X_train  : Training features DataFrame
        X_test   : Test features DataFrame
        y_train  : Training target Series
        y_test   : Test target Series
        alpha    : L1 regularization strength (default: 0.1)
        max_iter : Maximum iterations for convergence (default: 10000)
        scale    : Whether to StandardScale features (default: True)

    Returns:
        Dictionary with model, predictions, coefficients, and metrics
    """
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", Lasso(alpha=alpha, max_iter=max_iter)))

    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)

    model        = pipe.named_steps["model"]
    y_pred_train = pipe.predict(X_train)
    y_pred_test  = pipe.predict(X_test)
    metrics      = _evaluate(y_train, y_pred_train, y_test, y_pred_test)
    coefs        = pd.Series(model.coef_, index=X_train.columns)
    zero_feats   = coefs[coefs == 0].index.tolist()
    active_feats = coefs[coefs != 0].index.tolist()

    print(f"[Lasso] alpha={alpha} | max_iter={max_iter}")
    print(f"  Total features   : {len(X_train.columns)}")
    print(f"  Active (non-zero): {len(active_feats)}  → {active_feats}")
    print(f"  Zeroed out       : {len(zero_feats)}   → {zero_feats}")
    _print_metrics(metrics)

    return {
        "pipeline"      : pipe,
        "model"         : model,
        "y_pred_train"  : y_pred_train,
        "y_pred_test"   : y_pred_test,
        "metrics"       : metrics,
        "coefficients"  : coefs,
        "active_features": active_feats,
        "zero_features" : zero_feats,
        "sparsity_pct"  : round(len(zero_feats) / len(X_train.columns) * 100, 2),
        "alpha"         : alpha,
    }


# =============================================================================
# 🔧 2. LASSOCV — AUTO ALPHA SELECTION
# =============================================================================

def train_lasso_cv(X_train: pd.DataFrame,
                    X_test: pd.DataFrame,
                    y_train: pd.Series,
                    y_test: pd.Series,
                    alphas: list = None,
                    cv: int = 5,
                    max_iter: int = 10000) -> dict:
    """
    Trains Lasso with built-in cross-validation to select the best alpha.

    LassoCV fits the model across all alphas efficiently using the
    warm-start technique — more efficient than GridSearchCV for Lasso.

    Args:
        X_train  : Training features DataFrame
        X_test   : Test features DataFrame
        y_train  : Training target Series
        y_test   : Test target Series
        alphas   : List of alpha values (default: auto log-scale)
        cv       : Number of CV folds (default: 5)
        max_iter : Max iterations (default: 10000)

    Returns:
        Dictionary with best alpha, model, and metrics
    """
    scaler      = StandardScaler()
    X_tr_sc     = scaler.fit_transform(X_train)
    X_te_sc     = scaler.transform(X_test)

    model = LassoCV(
        alphas=alphas,
        cv=cv,
        max_iter=max_iter,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_tr_sc, y_train)

    best_alpha   = model.alpha_
    y_pred_train = model.predict(X_tr_sc)
    y_pred_test  = model.predict(X_te_sc)
    metrics      = _evaluate(y_train, y_pred_train, y_test, y_pred_test)
    coefs        = pd.Series(model.coef_, index=X_train.columns)
    zero_feats   = coefs[coefs == 0].index.tolist()
    active_feats = coefs[coefs != 0].index.tolist()

    print(f"[LassoCV] Best alpha: {best_alpha:.6f} | cv={cv}")
    print(f"  Active features : {len(active_feats)}")
    print(f"  Zeroed features : {len(zero_feats)}")
    _print_metrics(metrics)

    return {
        "model"         : model,
        "scaler"        : scaler,
        "best_alpha"    : best_alpha,
        "y_pred_train"  : y_pred_train,
        "y_pred_test"   : y_pred_test,
        "metrics"       : metrics,
        "coefficients"  : coefs,
        "active_features": active_feats,
        "zero_features" : zero_feats,
    }


# =============================================================================
# 🔧 3. GRIDSEARCHCV — LASSO ALPHA TUNING
# =============================================================================

def tune_lasso_alpha(X_train: pd.DataFrame,
                      y_train: pd.Series,
                      alphas: list = None,
                      cv: int = 5,
                      scoring: str = "r2") -> dict:
    """
    Tunes Lasso alpha using GridSearchCV with K-Fold cross-validation.

    Args:
        X_train : Training features DataFrame
        y_train : Training target Series
        alphas  : List of alpha values to search (default: log-scale)
        cv      : Number of CV folds (default: 5)
        scoring : Scoring metric (default: 'r2')

    Returns:
        Dictionary with best alpha, best score, and GridSearchCV object
    """
    if alphas is None:
        alphas = np.logspace(-4, 2, 30).tolist()

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Lasso(max_iter=10000))
    ])

    grid = GridSearchCV(
        pipe,
        param_grid={"model__alpha": alphas},
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best_alpha = grid.best_params_["model__alpha"]
    best_score = grid.best_score_

    print(f"[GridSearchCV Lasso] Best alpha: {best_alpha:.6f} | "
          f"Best CV {scoring}: {best_score:.4f}")

    return {
        "grid"       : grid,
        "best_alpha" : best_alpha,
        "best_score" : best_score,
        "best_model" : grid.best_estimator_,
        "cv_results" : pd.DataFrame(grid.cv_results_),
    }


# =============================================================================
# 🔧 4. ALPHA SENSITIVITY ANALYSIS
# =============================================================================

def alpha_sensitivity(X_train: pd.DataFrame,
                       X_test: pd.DataFrame,
                       y_train: pd.Series,
                       y_test: pd.Series,
                       alphas: list = None) -> pd.DataFrame:
    """
    Evaluates Lasso across a range of alpha values — shows how sparsity
    and performance change with regularization strength.

    Args:
        X_train : Training features DataFrame
        X_test  : Test features DataFrame
        y_train : Training target Series
        y_test  : Test target Series
        alphas  : List of alpha values to evaluate

    Returns:
        DataFrame with metrics, sparsity, and active feature count per alpha
    """
    if alphas is None:
        alphas = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    rows = []
    for a in alphas:
        model = LinearRegression() if a == 0.0 else Lasso(alpha=a, max_iter=10000)
        model.fit(X_tr_sc, y_train)

        y_tr_p  = model.predict(X_tr_sc)
        y_te_p  = model.predict(X_te_sc)
        coefs   = model.coef_
        n_zero  = (coefs == 0).sum()
        n_total = len(coefs)

        rows.append({
            "Alpha"         : a,
            "Train R²"      : round(r2_score(y_train, y_tr_p), 4),
            "Test R²"       : round(r2_score(y_test,  y_te_p), 4),
            "RMSE"          : round(np.sqrt(mean_squared_error(y_test, y_te_p)), 4),
            "Active Feats"  : int(n_total - n_zero),
            "Zero Feats"    : int(n_zero),
            "Sparsity %"    : round(n_zero / n_total * 100, 1),
            "Coef L1 Norm"  : round(np.sum(np.abs(coefs)), 4),
        })

    df = pd.DataFrame(rows)
    print("Lasso Alpha Sensitivity Analysis:")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 5. COEFFICIENT PATH (REGULARIZATION PATH)
# =============================================================================

def coefficient_path(X_train: pd.DataFrame,
                      y_train: pd.Series,
                      alphas: list = None) -> pd.DataFrame:
    """
    Computes the Lasso coefficient path — how each feature's coefficient
    changes (and when it reaches zero) as alpha increases.

    The path reveals:
      - Feature importance ranking (last to zero = most important)
      - Which features are truly relevant vs noise
      - Natural groupings of correlated features

    Args:
        X_train : Training features DataFrame
        y_train : Training target Series
        alphas  : List of alpha values (default: log-scale)

    Returns:
        DataFrame with coefficient values per feature per alpha value
    """
    if alphas is None:
        alphas = np.logspace(-4, 2, 60).tolist()

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_train)

    path_dict = {}
    for a in alphas:
        model = Lasso(alpha=a, max_iter=20000)
        model.fit(X_sc, y_train)
        path_dict[a] = dict(zip(X_train.columns, model.coef_))

    path_df = pd.DataFrame(path_dict).T
    path_df.index.name = "Alpha"

    # Report the alpha at which each feature first becomes zero
    print("[Lasso Coefficient Path]")
    for feat in X_train.columns:
        for a, val in path_dict.items():
            if val[feat] == 0.0:
                print(f"  '{feat}' → zeroed at alpha ≈ {a:.6f}")
                break
        else:
            print(f"  '{feat}' → never zeroed in this alpha range")

    return path_df


# =============================================================================
# 🔧 6. FEATURE SELECTION VIA LASSO
# =============================================================================

def select_features_lasso(X_train: pd.DataFrame,
                            X_test: pd.DataFrame,
                            y_train: pd.Series,
                            y_test: pd.Series,
                            alpha: float = None) -> dict:
    """
    Uses Lasso as an automatic feature selector:
      1. Fit Lasso (with LassoCV if alpha not provided)
      2. Extract features with non-zero coefficients
      3. Refit a clean OLS on selected features

    Two-stage approach: Lasso for selection → OLS for interpretation.

    Args:
        X_train : Training features DataFrame
        X_test  : Test features DataFrame
        y_train : Training target Series
        y_test  : Test target Series
        alpha   : Fixed alpha (default: auto via LassoCV)

    Returns:
        Dictionary with selected features and refitted OLS metrics
    """
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    # Step 1: Fit Lasso (auto or fixed alpha)
    if alpha is None:
        lasso = LassoCV(cv=5, max_iter=10000, random_state=42, n_jobs=-1)
    else:
        lasso = Lasso(alpha=alpha, max_iter=10000)

    lasso.fit(X_tr_sc, y_train)
    best_alpha = lasso.alpha_ if alpha is None else alpha

    coefs         = pd.Series(lasso.coef_, index=X_train.columns)
    selected_cols = coefs[coefs != 0].index.tolist()
    dropped_cols  = coefs[coefs == 0].index.tolist()

    print(f"[Lasso Feature Selection] alpha={best_alpha:.6f}")
    print(f"  Selected : {len(selected_cols)} features → {selected_cols}")
    print(f"  Dropped  : {len(dropped_cols)}  features → {dropped_cols}")

    # Step 2: Refit OLS on selected features only
    if len(selected_cols) == 0:
        print("  ⚠️  All features zeroed out — increase alpha range")
        return {"selected": [], "dropped": dropped_cols}

    ols_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression())
    ])
    ols_pipe.fit(X_train[selected_cols], y_train)
    y_ols_pred = ols_pipe.predict(X_test[selected_cols])

    ols_metrics = {
        "R²"  : round(r2_score(y_test, y_ols_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_ols_pred)), 4),
        "MAE" : round(mean_absolute_error(y_test, y_ols_pred), 4),
    }

    print(f"\n  OLS on selected features — Test Metrics:")
    for k, v in ols_metrics.items():
        print(f"    {k}: {v}")

    return {
        "lasso_alpha"   : best_alpha,
        "selected"      : selected_cols,
        "dropped"       : dropped_cols,
        "sparsity_pct"  : round(len(dropped_cols) / len(X_train.columns) * 100, 2),
        "ols_pipeline"  : ols_pipe,
        "ols_metrics"   : ols_metrics,
        "lasso_coefs"   : coefs,
    }


# =============================================================================
# 🔧 7. COMPARE LASSO VS RIDGE VS LINEAR
# =============================================================================

def compare_models(X_train: pd.DataFrame,
                    X_test: pd.DataFrame,
                    y_train: pd.Series,
                    y_test: pd.Series,
                    lasso_alpha: float = 0.1,
                    ridge_alpha: float = 1.0) -> pd.DataFrame:
    """
    Side-by-side comparison of Linear Regression, Ridge, and Lasso.

    Args:
        X_train     : Training features DataFrame
        X_test      : Test features DataFrame
        y_train     : Training target Series
        y_test      : Test target Series
        lasso_alpha : Alpha for Lasso model
        ridge_alpha : Alpha for Ridge model

    Returns:
        DataFrame with comparison metrics
    """
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    configs = [
        ("Linear Regression", LinearRegression()),
        (f"Ridge (α={ridge_alpha})", Ridge(alpha=ridge_alpha)),
        (f"Lasso (α={lasso_alpha})", Lasso(alpha=lasso_alpha, max_iter=10000)),
    ]

    rows = []
    for name, model in configs:
        model.fit(X_tr_sc, y_train)
        y_tr = model.predict(X_tr_sc)
        y_te = model.predict(X_te_sc)
        coefs  = model.coef_
        n_zero = int((coefs == 0).sum())

        rows.append({
            "Model"        : name,
            "Train R²"     : round(r2_score(y_train, y_tr), 4),
            "Test R²"      : round(r2_score(y_test,  y_te), 4),
            "RMSE"         : round(np.sqrt(mean_squared_error(y_test, y_te)), 4),
            "MAE"          : round(mean_absolute_error(y_test, y_te), 4),
            "Zero Coefs"   : n_zero,
            "Coef L1 Norm" : round(np.sum(np.abs(coefs)), 4),
        })

    comp_df = pd.DataFrame(rows)
    print("Model Comparison — Linear vs Ridge vs Lasso:")
    print(comp_df.to_string(index=False))
    return comp_df


# =============================================================================
# 🔧 8. CROSS-VALIDATION
# =============================================================================

def cross_validate_lasso(X: pd.DataFrame,
                           y: pd.Series,
                           alpha: float = 0.1,
                           cv: int = 5,
                           scoring: str = "r2") -> dict:
    """
    Performs K-Fold Cross-Validation on Lasso Regression Pipeline.

    Args:
        X       : Full feature DataFrame
        y       : Full target Series
        alpha   : Lasso regularization strength (default: 0.1)
        cv      : Number of folds (default: 5)
        scoring : Scoring metric (default: 'r2')

    Returns:
        Dictionary with fold scores, mean, and std
    """
    pipe   = Pipeline([("scaler", StandardScaler()),
                        ("model",  Lasso(alpha=alpha, max_iter=10000))])
    kf     = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=kf, scoring=scoring)

    result = {
        "scores"      : scores,
        "mean"        : scores.mean(),
        "std"         : scores.std(),
        "fold_results": {f"Fold {i+1}": round(s, 4) for i, s in enumerate(scores)},
    }

    print(f"[Lasso CV] alpha={alpha} | K={cv} | "
          f"{scoring.upper()}: {scores.mean():.4f} ± {scores.std():.4f}")
    for fold, score in result["fold_results"].items():
        print(f"  {fold}: {score}")

    return result


# =============================================================================
# 🔧 9. EVALUATION METRICS
# =============================================================================

def evaluate_lasso(y_train, y_pred_train, y_test, y_pred_test) -> dict:
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
    return evaluate_lasso(y_train, y_pred_train, y_test, y_pred_test)


def _print_metrics(metrics: dict) -> None:
    for split in ["train", "test"]:
        m = metrics[split]
        print(f"  [{split.upper():5s}] MAE={m['MAE']:>10.4f} | "
              f"RMSE={m['RMSE']:>10.4f} | R²={m['R²']:>7.4f}")


# =============================================================================
# 🚀 MAIN — Demo with Synthetic High-Dimensional Dataset
# =============================================================================

if __name__ == "__main__":

    # ── Synthetic dataset: 20 features, only 5 are truly relevant ─────────
    np.random.seed(42)
    n          = 400
    n_features = 20
    n_relevant = 5

    X_data = np.random.randn(n, n_features)
    true_coefs = np.zeros(n_features)
    true_coefs[:n_relevant] = [3.0, -2.5, 1.8, -1.2, 0.9]   # only first 5 matter

    y_data = X_data @ true_coefs + np.random.normal(0, 1, n)

    feat_names = [f"Feature_{i+1:02d}" for i in range(n_features)]
    X = pd.DataFrame(X_data, columns=feat_names)
    y = pd.Series(y_data, name="Target")

    print("=" * 65)
    print("📊 Dataset Info — High-Dimensional Sparse Problem")
    print("=" * 65)
    print(f"Shape    : {X.shape}")
    print(f"Relevant : {n_relevant} features")
    print(f"Noise    : {n_features - n_relevant} irrelevant features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── 1. Lasso fixed alpha ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("1️⃣  Lasso Regression (alpha=0.1)")
    print("=" * 65)
    result = train_lasso(X_train, X_test, y_train, y_test, alpha=0.1)

    # ── 2. LassoCV ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("2️⃣  LassoCV — Auto Alpha Selection")
    print("=" * 65)
    cv_result = train_lasso_cv(X_train, X_test, y_train, y_test)

    # ── 3. GridSearchCV ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("3️⃣  GridSearchCV — Lasso Alpha Tuning")
    print("=" * 65)
    gs_result = tune_lasso_alpha(X_train, y_train)

    # ── 4. Alpha Sensitivity ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("4️⃣  Alpha Sensitivity Analysis")
    print("=" * 65)
    sens_df = alpha_sensitivity(X_train, X_test, y_train, y_test)

    # ── 5. Coefficient Path ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("5️⃣  Lasso Coefficient Path")
    print("=" * 65)
    path_df = coefficient_path(X_train, y_train)

    # ── 6. Feature Selection ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("6️⃣  Lasso as Feature Selector → OLS on selected features")
    print("=" * 65)
    sel_result = select_features_lasso(
        X_train, X_test, y_train, y_test,
        alpha=cv_result["best_alpha"]
    )

    # ── 7. Model Comparison ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("7️⃣  Linear vs Ridge vs Lasso")
    print("=" * 65)
    compare_models(
        X_train, X_test, y_train, y_test,
        lasso_alpha=cv_result["best_alpha"],
        ridge_alpha=1.0
    )

    # ── 8. Cross-Validation ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("8️⃣  5-Fold Cross-Validation")
    print("=" * 65)
    cross_validate_lasso(X, y, alpha=cv_result["best_alpha"], cv=5)

    print("\n✅ All Lasso Regression techniques demonstrated successfully!")
