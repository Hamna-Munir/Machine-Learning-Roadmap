# =============================================================================
# 📦 Ridge Regression — Reusable ML Script
# =============================================================================
# Author   : Hamna Munir
# Topic    : 03_Supervised_Learning / Regression / Ridge_Regression
# File     : ridge_regression.py
# =============================================================================

# -----------------------------------------------------------------------------
# 📚 Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    KFold, GridSearchCV, validation_curve
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    r2_score, mean_absolute_percentage_error
)
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 🔧 1. TRAIN RIDGE REGRESSION (Fixed Alpha)
# =============================================================================

def train_ridge(X_train: pd.DataFrame,
                 X_test: pd.DataFrame,
                 y_train: pd.Series,
                 y_test: pd.Series,
                 alpha: float = 1.0,
                 scale: bool = True) -> dict:
    """
    Trains a Ridge Regression model with a fixed regularization alpha.

    Formula:
        Loss = RSS + alpha × Σβⱼ²

    Best for:
        - Multicollinear features
        - When all features should be retained (no elimination)
        - Preventing overfitting in linear models

    Args:
        X_train : Training features DataFrame
        X_test  : Test features DataFrame
        y_train : Training target Series
        y_test  : Test target Series
        alpha   : L2 regularization strength (default: 1.0)
        scale   : Whether to StandardScale features (default: True)

    Returns:
        Dictionary with model, predictions, coefficients, and metrics
    """
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", Ridge(alpha=alpha)))

    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)

    y_pred_train = pipe.predict(X_train)
    y_pred_test  = pipe.predict(X_test)

    metrics = _evaluate(y_train, y_pred_train, y_test, y_pred_test)

    model  = pipe.named_steps["model"]
    coefs  = pd.Series(model.coef_, index=X_train.columns).sort_values()

    print(f"[Ridge] alpha={alpha} | scale={scale}")
    print(f"  Intercept    : {model.intercept_:.4f}")
    _print_metrics(metrics)

    return {
        "pipeline"     : pipe,
        "model"        : model,
        "y_pred_train" : y_pred_train,
        "y_pred_test"  : y_pred_test,
        "metrics"      : metrics,
        "coefficients" : coefs,
        "intercept"    : model.intercept_,
        "alpha"        : alpha,
    }


# =============================================================================
# 🔧 2. RIDGECV — BUILT-IN CROSS-VALIDATED ALPHA SELECTION
# =============================================================================

def train_ridge_cv(X_train: pd.DataFrame,
                    X_test: pd.DataFrame,
                    y_train: pd.Series,
                    y_test: pd.Series,
                    alphas: list = None,
                    cv: int = 5) -> dict:
    """
    Trains Ridge Regression with built-in cross-validation to select alpha.

    RidgeCV is more efficient than GridSearchCV for alpha selection —
    it uses Leave-One-Out CV by default or K-Fold when cv is specified.

    Args:
        X_train : Training features DataFrame
        X_test  : Test features DataFrame
        y_train : Training target Series
        y_test  : Test target Series
        alphas  : List of alpha values to try (default: log-scale range)
        cv      : Number of CV folds (default: 5)

    Returns:
        Dictionary with best alpha, model, predictions, and metrics
    """
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RidgeCV(alphas=alphas, cv=cv, scoring="r2"))
    ])
    pipe.fit(X_train, y_train)

    model        = pipe.named_steps["model"]
    best_alpha   = model.alpha_
    y_pred_train = pipe.predict(X_train)
    y_pred_test  = pipe.predict(X_test)
    metrics      = _evaluate(y_train, y_pred_train, y_test, y_pred_test)

    print(f"[RidgeCV] Best alpha: {best_alpha} | cv={cv}")
    print(f"  Searched alphas: {alphas}")
    _print_metrics(metrics)

    return {
        "pipeline"     : pipe,
        "model"        : model,
        "best_alpha"   : best_alpha,
        "y_pred_train" : y_pred_train,
        "y_pred_test"  : y_pred_test,
        "metrics"      : metrics,
        "coefficients" : pd.Series(model.coef_, index=X_train.columns),
    }


# =============================================================================
# 🔧 3. GRIDSEARCHCV — RIDGE ALPHA TUNING
# =============================================================================

def tune_ridge_alpha(X_train: pd.DataFrame,
                      y_train: pd.Series,
                      alphas: list = None,
                      cv: int = 5,
                      scoring: str = "r2") -> dict:
    """
    Tunes Ridge alpha using GridSearchCV with K-Fold cross-validation.

    Args:
        X_train : Training features DataFrame
        y_train : Training target Series
        alphas  : List of alpha values to search (default: log-scale)
        cv      : Number of CV folds (default: 5)
        scoring : Scoring metric (default: 'r2')

    Returns:
        Dictionary with best alpha, best score, and fitted GridSearchCV
    """
    if alphas is None:
        alphas = np.logspace(-3, 4, 20).tolist()

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge())
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

    print(f"[GridSearchCV Ridge] Best alpha: {best_alpha:.6f} | "
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
    Evaluates Ridge performance across a range of alpha values.

    Shows how train/test R², RMSE, and coefficient norms change with alpha.

    Args:
        X_train : Training features DataFrame
        X_test  : Test features DataFrame
        y_train : Training target Series
        y_test  : Test target Series
        alphas  : List of alpha values to evaluate

    Returns:
        DataFrame with metrics per alpha value
    """
    if alphas is None:
        alphas = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    rows = []
    for a in alphas:
        if a == 0.0:
            model = LinearRegression()
        else:
            model = Ridge(alpha=a)
        model.fit(X_tr_sc, y_train)

        y_tr_pred = model.predict(X_tr_sc)
        y_te_pred = model.predict(X_te_sc)

        rows.append({
            "Alpha"        : a,
            "Train R²"     : round(r2_score(y_train, y_tr_pred), 4),
            "Test R²"      : round(r2_score(y_test,  y_te_pred), 4),
            "Train RMSE"   : round(np.sqrt(mean_squared_error(y_train, y_tr_pred)), 4),
            "Test RMSE"    : round(np.sqrt(mean_squared_error(y_test,  y_te_pred)), 4),
            "Coef L2 Norm" : round(np.sqrt(np.sum(model.coef_ ** 2)), 4),
            "Max |Coef|"   : round(np.max(np.abs(model.coef_)), 4),
        })

    df = pd.DataFrame(rows)
    print("Alpha Sensitivity Analysis:")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 5. COEFFICIENT PATH (REGULARIZATION PATH)
# =============================================================================

def coefficient_path(X_train: pd.DataFrame,
                      y_train: pd.Series,
                      alphas: list = None) -> pd.DataFrame:
    """
    Computes how each feature's coefficient changes as alpha increases.

    The regularization path shows the L2 shrinkage effect:
    - All coefficients start at OLS values (alpha=0)
    - All coefficients shrink toward 0 as alpha → ∞
    - No coefficient reaches exactly 0

    Args:
        X_train : Training features DataFrame
        y_train : Training target Series
        alphas  : List of alpha values (default: log-scale)

    Returns:
        DataFrame with coefficients per feature per alpha value
    """
    if alphas is None:
        alphas = np.logspace(-3, 4, 50).tolist()

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)

    coef_rows = {}
    for a in alphas:
        model = Ridge(alpha=a).fit(X_tr_sc, y_train)
        coef_rows[a] = dict(zip(X_train.columns, model.coef_))

    path_df = pd.DataFrame(coef_rows).T
    path_df.index.name = "Alpha"

    print(f"[Coefficient Path] {len(alphas)} alpha values | "
          f"{len(X_train.columns)} features")
    print(f"  Alpha range: [{min(alphas):.4f}, {max(alphas):.0f}]")

    return path_df


# =============================================================================
# 🔧 6. COMPARE RIDGE VS LINEAR REGRESSION
# =============================================================================

def compare_ridge_vs_linear(X_train: pd.DataFrame,
                              X_test: pd.DataFrame,
                              y_train: pd.Series,
                              y_test: pd.Series,
                              alpha: float = 1.0) -> pd.DataFrame:
    """
    Side-by-side comparison of Linear Regression vs Ridge Regression.

    Args:
        X_train : Training features DataFrame
        X_test  : Test features DataFrame
        y_train : Training target Series
        y_test  : Test target Series
        alpha   : Ridge alpha for comparison (default: 1.0)

    Returns:
        DataFrame with comparison metrics for both models
    """
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        f"Ridge (α={alpha})": Ridge(alpha=alpha),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_tr_sc, y_train)
        y_tr = model.predict(X_tr_sc)
        y_te = model.predict(X_te_sc)
        rows.append({
            "Model"        : name,
            "Train R²"     : round(r2_score(y_train, y_tr), 4),
            "Test R²"      : round(r2_score(y_test,  y_te), 4),
            "Train RMSE"   : round(np.sqrt(mean_squared_error(y_train, y_tr)), 4),
            "Test RMSE"    : round(np.sqrt(mean_squared_error(y_test,  y_te)), 4),
            "MAE"          : round(mean_absolute_error(y_test, y_te), 4),
            "Coef L2 Norm" : round(np.sqrt(np.sum(model.coef_**2)), 4),
        })

    comp_df = pd.DataFrame(rows)
    print("Ridge vs Linear Regression Comparison:")
    print(comp_df.to_string(index=False))
    return comp_df


# =============================================================================
# 🔧 7. CROSS-VALIDATION
# =============================================================================

def cross_validate_ridge(X: pd.DataFrame,
                           y: pd.Series,
                           alpha: float = 1.0,
                           cv: int = 5,
                           scoring: str = "r2") -> dict:
    """
    Performs K-Fold Cross-Validation on Ridge Regression Pipeline.

    Args:
        X       : Full feature DataFrame
        y       : Full target Series
        alpha   : Ridge regularization strength (default: 1.0)
        cv      : Number of folds (default: 5)
        scoring : Scoring metric (default: 'r2')

    Returns:
        Dictionary with fold scores, mean, and std
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=alpha))
    ])
    kf     = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=kf, scoring=scoring)

    result = {
        "scores"      : scores,
        "mean"        : scores.mean(),
        "std"         : scores.std(),
        "fold_results": {f"Fold {i+1}": round(s, 4) for i, s in enumerate(scores)},
    }

    print(f"[Ridge CV] alpha={alpha} | K={cv} | "
          f"{scoring.upper()}: {scores.mean():.4f} ± {scores.std():.4f}")
    for fold, score in result["fold_results"].items():
        print(f"  {fold}: {score}")

    return result


# =============================================================================
# 🔧 8. EVALUATION METRICS
# =============================================================================

def evaluate_ridge(y_train: pd.Series,
                    y_pred_train: np.ndarray,
                    y_test: pd.Series,
                    y_pred_test: np.ndarray) -> dict:
    """
    Computes regression evaluation metrics for train and test sets.

    Metrics: MAE, MSE, RMSE, MAPE, R²

    Args:
        y_train      : True training labels
        y_pred_train : Predicted training values
        y_test       : True test labels
        y_pred_test  : Predicted test values

    Returns:
        Dictionary with metrics for both splits
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

    return {
        "train": metrics(y_train, y_pred_train),
        "test" : metrics(y_test,  y_pred_test),
    }


# =============================================================================
# 🔧 HELPERS
# =============================================================================

def _evaluate(y_train, y_pred_train, y_test, y_pred_test):
    return evaluate_ridge(y_train, y_pred_train, y_test, y_pred_test)


def _print_metrics(metrics: dict) -> None:
    for split in ["train", "test"]:
        m = metrics[split]
        print(f"  [{split.upper():5s}] MAE={m['MAE']:>10.4f} | "
              f"RMSE={m['RMSE']:>10.4f} | R²={m['R²']:>7.4f}")


# =============================================================================
# 🚀 MAIN — Demo with Synthetic Multicollinear Dataset
# =============================================================================

if __name__ == "__main__":

    # ── Synthetic multicollinear dataset ──────────────────────────────────
    np.random.seed(42)
    n = 500

    base    = np.random.normal(0, 1, n)
    X = pd.DataFrame({
        "Feature_1" : base + np.random.normal(0, 0.1, n),   # highly correlated
        "Feature_2" : base + np.random.normal(0, 0.1, n),   # highly correlated
        "Feature_3" : np.random.normal(0, 1, n),             # independent
        "Feature_4" : np.random.normal(0, 1, n),             # independent
        "Feature_5" : base * 0.8 + np.random.normal(0, 0.2, n),  # correlated
    })

    y = pd.Series(
        3.0 * X["Feature_1"]
        + 2.5 * X["Feature_2"]
        - 1.5 * X["Feature_3"]
        + 0.8 * X["Feature_4"]
        + np.random.normal(0, 0.5, n),
        name="Target"
    )

    print("=" * 65)
    print("📊 Dataset Info — Multicollinear Features")
    print("=" * 65)
    print(f"Shape  : {X.shape}")
    print(f"Target : {y.name}  |  Mean: {y.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── 1. Ridge with fixed alpha ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("1️⃣  Ridge Regression (alpha=1.0)")
    print("=" * 65)
    result = train_ridge(X_train, X_test, y_train, y_test, alpha=1.0)

    # ── 2. RidgeCV ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("2️⃣  RidgeCV — Auto Alpha Selection")
    print("=" * 65)
    cv_result = train_ridge_cv(X_train, X_test, y_train, y_test)

    # ── 3. GridSearchCV ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("3️⃣  GridSearchCV — Ridge Alpha Tuning")
    print("=" * 65)
    gs_result = tune_ridge_alpha(X_train, y_train)

    # ── 4. Alpha Sensitivity ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("4️⃣  Alpha Sensitivity Analysis")
    print("=" * 65)
    sensitivity_df = alpha_sensitivity(X_train, X_test, y_train, y_test)

    # ── 5. Coefficient Path ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("5️⃣  Coefficient Regularization Path")
    print("=" * 65)
    path_df = coefficient_path(X_train, y_train)
    print(path_df.head(5).round(4).to_string())

    # ── 6. Ridge vs Linear ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("6️⃣  Ridge vs Linear Regression")
    print("=" * 65)
    compare_ridge_vs_linear(
        X_train, X_test, y_train, y_test,
        alpha=cv_result["best_alpha"]
    )

    # ── 7. Cross-Validation ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("7️⃣  5-Fold Cross-Validation")
    print("=" * 65)
    cross_validate_ridge(X, y, alpha=cv_result["best_alpha"], cv=5)

    print("\n✅ All Ridge Regression techniques demonstrated successfully!")
