# =============================================================================
# 📦 Polynomial Regression — Reusable ML Script
# =============================================================================
# Author   : Hamna Munir
# Topic    : 03_Supervised_Learning / Regression / Polynomial_Regression
# File     : polynomial_regression.py
# =============================================================================

# -----------------------------------------------------------------------------
# 📚 Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    KFold, validation_curve
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    r2_score, mean_absolute_percentage_error
)
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 🔧 1. BUILD POLYNOMIAL REGRESSION PIPELINE
# =============================================================================

def build_poly_pipeline(degree: int = 2,
                          interaction_only: bool = False,
                          include_bias: bool = False,
                          scale: bool = True,
                          regularize: bool = False,
                          alpha: float = 1.0) -> Pipeline:
    """
    Builds a full Polynomial Regression sklearn Pipeline.

    Pipeline steps:
        PolynomialFeatures → (StandardScaler) → LinearRegression or Ridge

    Args:
        degree           : Polynomial degree (default: 2)
        interaction_only : Only cross-product terms, no powers (default: False)
        include_bias     : Include a bias/constant column (default: False)
        scale            : Apply StandardScaler after feature expansion (default: True)
        regularize       : Use Ridge instead of Linear Regression (default: False)
        alpha            : Ridge regularization strength (default: 1.0)

    Returns:
        sklearn Pipeline object

    Example:
        pipe = build_poly_pipeline(degree=3, regularize=True, alpha=0.5)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
    """
    steps = [
        ("poly",   PolynomialFeatures(degree=degree,
                                       interaction_only=interaction_only,
                                       include_bias=include_bias)),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))

    if regularize:
        steps.append(("model", Ridge(alpha=alpha)))
    else:
        steps.append(("model", LinearRegression()))

    pipeline = Pipeline(steps)
    reg_str  = f"Ridge(α={alpha})" if regularize else "LinearRegression"
    print(f"[Pipeline] PolynomialFeatures(degree={degree}) → "
          f"{'StandardScaler → ' if scale else ''}{reg_str}")
    return pipeline


# =============================================================================
# 🔧 2. TRAIN POLYNOMIAL REGRESSION
# =============================================================================

def train_poly_regression(X_train: pd.DataFrame,
                            X_test: pd.DataFrame,
                            y_train: pd.Series,
                            y_test: pd.Series,
                            degree: int = 2,
                            scale: bool = True,
                            regularize: bool = False,
                            alpha: float = 1.0) -> dict:
    """
    Trains a Polynomial Regression model and returns evaluation results.

    Args:
        X_train    : Training features
        X_test     : Test features
        y_train    : Training target
        y_test     : Test target
        degree     : Polynomial degree (default: 2)
        scale      : Apply StandardScaler (default: True)
        regularize : Use Ridge regularization (default: False)
        alpha      : Ridge alpha (default: 1.0)

    Returns:
        Dictionary with pipeline, predictions, and metrics
    """
    pipeline = build_poly_pipeline(
        degree=degree, scale=scale,
        regularize=regularize, alpha=alpha
    )
    pipeline.fit(X_train, y_train)

    y_pred_train = pipeline.predict(X_train)
    y_pred_test  = pipeline.predict(X_test)

    metrics = _compute_metrics(y_train, y_pred_train, y_test, y_pred_test)

    # Number of features after expansion
    poly_step    = pipeline.named_steps["poly"]
    n_poly_feats = poly_step.transform(X_train[:1]).shape[1]

    print(f"\n[Poly Degree={degree}] "
          f"Features: {X_train.shape[1]} → {n_poly_feats}")
    _print_metrics(metrics)

    return {
        "pipeline"      : pipeline,
        "y_pred_train"  : y_pred_train,
        "y_pred_test"   : y_pred_test,
        "metrics"       : metrics,
        "n_poly_feats"  : n_poly_feats,
        "degree"        : degree,
    }


# =============================================================================
# 🔧 3. COMPARE MULTIPLE DEGREES
# =============================================================================

def compare_degrees(X_train: pd.DataFrame,
                     X_test: pd.DataFrame,
                     y_train: pd.Series,
                     y_test: pd.Series,
                     degrees: list = None,
                     scale: bool = True) -> pd.DataFrame:
    """
    Trains Polynomial Regression for multiple degrees and compares performance.

    Reveals the bias-variance tradeoff — underfitting at low degrees,
    overfitting at high degrees.

    Args:
        X_train : Training features
        X_test  : Test features
        y_train : Training target
        y_test  : Test target
        degrees : List of degree values to compare (default: [1, 2, 3, 4, 5])
        scale   : Apply StandardScaler (default: True)

    Returns:
        DataFrame with train/test metrics for each degree
    """
    if degrees is None:
        degrees = [1, 2, 3, 4, 5]

    rows = []
    for d in degrees:
        try:
            pipe = build_poly_pipeline(degree=d, scale=scale)
            pipe.fit(X_train, y_train)
            y_tr_pred = pipe.predict(X_train)
            y_te_pred = pipe.predict(X_test)

            n_feats = pipe.named_steps["poly"].transform(X_train[:1]).shape[1]
            r2_tr   = r2_score(y_train, y_tr_pred)
            r2_te   = r2_score(y_test,  y_te_pred)

            rows.append({
                "Degree"         : d,
                "Poly Features"  : n_feats,
                "Train R²"       : round(r2_tr, 4),
                "Test R²"        : round(r2_te, 4),
                "Train RMSE"     : round(np.sqrt(mean_squared_error(y_train, y_tr_pred)), 2),
                "Test RMSE"      : round(np.sqrt(mean_squared_error(y_test,  y_te_pred)), 2),
                "Train MAE"      : round(mean_absolute_error(y_train, y_tr_pred), 2),
                "Test MAE"       : round(mean_absolute_error(y_test,  y_te_pred), 2),
                "Overfit Gap"    : round(r2_tr - r2_te, 4),
            })
        except Exception as e:
            print(f"  ⚠️  Degree={d} failed: {e}")

    df_result = pd.DataFrame(rows)
    print("\n[Degree Comparison]")
    print(df_result.to_string(index=False))
    return df_result


# =============================================================================
# 🔧 4. CROSS-VALIDATE ACROSS DEGREES
# =============================================================================

def cv_across_degrees(X: pd.DataFrame,
                       y: pd.Series,
                       degrees: list = None,
                       cv: int = 5,
                       scoring: str = "r2") -> pd.DataFrame:
    """
    Uses K-Fold Cross-Validation to find the optimal polynomial degree.

    This is the correct way to select degree — avoids overfitting to a
    single train/test split.

    Args:
        X       : Full feature DataFrame
        y       : Full target Series
        degrees : List of degrees to evaluate (default: [1, 2, 3, 4, 5])
        cv      : Number of CV folds (default: 5)
        scoring : Metric — 'r2', 'neg_mean_squared_error' (default: 'r2')

    Returns:
        DataFrame with mean and std CV score per degree
    """
    if degrees is None:
        degrees = [1, 2, 3, 4, 5]

    kf   = KFold(n_splits=cv, shuffle=True, random_state=42)
    rows = []

    for d in degrees:
        try:
            pipe   = build_poly_pipeline(degree=d, scale=True)
            scores = cross_val_score(pipe, X, y, cv=kf, scoring=scoring)
            rows.append({
                "Degree"  : d,
                f"CV Mean {scoring}" : round(scores.mean(), 4),
                f"CV Std  {scoring}" : round(scores.std(),  4),
                "CV Min"  : round(scores.min(), 4),
                "CV Max"  : round(scores.max(), 4),
            })
        except Exception as e:
            print(f"  ⚠️  Degree={d} CV failed: {e}")

    df_cv = pd.DataFrame(rows)
    best  = df_cv.loc[df_cv[f"CV Mean {scoring}"].idxmax(), "Degree"]

    print(f"\n[Cross-Validation Degree Selection] K={cv} | scoring={scoring}")
    print(df_cv.to_string(index=False))
    print(f"\n✅ Best degree by CV {scoring}: {best}")
    return df_cv


# =============================================================================
# 🔧 5. POLYNOMIAL FEATURES REPORT
# =============================================================================

def poly_features_report(X: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
    """
    Generates a report of all features created by PolynomialFeatures.

    Shows the mapping from original features to polynomial/interaction terms.

    Args:
        X      : Feature DataFrame
        degree : Polynomial degree

    Returns:
        DataFrame with feature names and their polynomial expansions
    """
    poly  = PolynomialFeatures(degree=degree, include_bias=False)
    poly.fit(X)
    names = poly.get_feature_names_out(X.columns)

    original   = [n for n in names if " " not in n and "^" not in n]
    powers     = [n for n in names if "^" in n]
    interact   = [n for n in names if " " in n and "^" not in n]

    report = pd.DataFrame({
        "Feature Name"  : names,
        "Type"          : [
            "Original"    if n in original else
            "Power Term"  if "^" in n else
            "Interaction"
            for n in names
        ],
    })

    print(f"\n[PolynomialFeatures] degree={degree} | "
          f"{X.shape[1]} original → {len(names)} features")
    print(f"  Original    : {len(original)}")
    print(f"  Power Terms : {len(powers)}")
    print(f"  Interactions: {len(interact)}")
    print(report.to_string(index=False))
    return report


# =============================================================================
# 🔧 6. RIDGE-REGULARIZED POLYNOMIAL REGRESSION
# =============================================================================

def train_ridge_poly(X_train: pd.DataFrame,
                      X_test: pd.DataFrame,
                      y_train: pd.Series,
                      y_test: pd.Series,
                      degree: int = 3,
                      alphas: list = None) -> pd.DataFrame:
    """
    Trains Ridge-regularized Polynomial Regression for multiple alpha values.

    Combines PolynomialFeatures with Ridge to prevent overfitting at higher degrees.
    Finds the best alpha via comparing test R².

    Args:
        X_train : Training features
        X_test  : Test features
        y_train : Training target
        y_test  : Test target
        degree  : Polynomial degree (default: 3)
        alphas  : List of Ridge alpha values to try

    Returns:
        DataFrame with metrics per alpha value
    """
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    rows = []
    for a in alphas:
        pipe = build_poly_pipeline(degree=degree, regularize=True, alpha=a)
        pipe.fit(X_train, y_train)
        y_tr_pred = pipe.predict(X_train)
        y_te_pred = pipe.predict(X_test)

        rows.append({
            "Alpha"     : a,
            "Train R²"  : round(r2_score(y_train, y_tr_pred), 4),
            "Test R²"   : round(r2_score(y_test,  y_te_pred), 4),
            "Train RMSE": round(np.sqrt(mean_squared_error(y_train, y_tr_pred)), 2),
            "Test RMSE" : round(np.sqrt(mean_squared_error(y_test,  y_te_pred)), 2),
        })

    df_result = pd.DataFrame(rows)
    best_a    = df_result.loc[df_result["Test R²"].idxmax(), "Alpha"]

    print(f"\n[Ridge Poly Degree={degree}] Alpha Search:")
    print(df_result.to_string(index=False))
    print(f"\n✅ Best alpha: {best_a}")
    return df_result


# =============================================================================
# 🔧 7. EVALUATE REGRESSION
# =============================================================================

def _compute_metrics(y_train, y_pred_train, y_test, y_pred_test) -> dict:
    """Computes MAE, RMSE, R² for train and test sets."""
    def _metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return {
            "MAE"  : round(mean_absolute_error(y_true, y_pred), 4),
            "RMSE" : round(np.sqrt(mse), 4),
            "MAPE" : round(mean_absolute_percentage_error(y_true, y_pred) * 100, 4),
            "R²"   : round(r2_score(y_true, y_pred), 4),
        }
    return {"train": _metrics(y_train, y_pred_train),
            "test" : _metrics(y_test,  y_pred_test)}


def _print_metrics(metrics: dict) -> None:
    """Internal helper to print train/test evaluation metrics."""
    for split in ["train", "test"]:
        m = metrics[split]
        print(f"  [{split.upper():5s}] "
              f"MAE={m['MAE']:>10.4f} | "
              f"RMSE={m['RMSE']:>10.4f} | "
              f"R²={m['R²']:>7.4f}")


# =============================================================================
# 🚀 MAIN — Demo with Synthetic Non-Linear Dataset
# =============================================================================

if __name__ == "__main__":

    # ── Synthetic Non-Linear Dataset ────────────────────────────────────────
    np.random.seed(42)
    n = 400

    X_raw = pd.DataFrame({
        "X1": np.linspace(-3, 3, n),
        "X2": np.random.uniform(-2, 2, n),
    })

    # True non-linear relationship
    y = pd.Series(
        2 * X_raw["X1"]**2
        - 3 * X_raw["X1"]
        + 1.5 * X_raw["X2"]**2
        + np.random.normal(0, 1.5, n),
        name="y"
    )

    print("=" * 65)
    print("📊 Dataset Info — Non-Linear Synthetic Data")
    print("=" * 65)
    print(f"Shape  : {X_raw.shape}")
    print(f"Target : y = 2X₁² − 3X₁ + 1.5X₂² + noise")

    # ── Train-Test Split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42
    )

    # ── 1. Polynomial Features Report ─────────────────────────────────────
    print("\n" + "=" * 65)
    print("1️⃣  Polynomial Feature Expansion Report (degree=2)")
    print("=" * 65)
    poly_features_report(X_raw, degree=2)

    # ── 2. Train degree=2 ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("2️⃣  Train Polynomial Regression (degree=2)")
    print("=" * 65)
    result_d2 = train_poly_regression(
        X_train, X_test, y_train, y_test, degree=2
    )

    # ── 3. Compare degrees ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("3️⃣  Degree Comparison (1 → 5)")
    print("=" * 65)
    degree_df = compare_degrees(
        X_train, X_test, y_train, y_test, degrees=[1, 2, 3, 4, 5]
    )

    # ── 4. CV across degrees ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("4️⃣  Cross-Validation Degree Selection")
    print("=" * 65)
    cv_df = cv_across_degrees(X_raw, y, degrees=[1, 2, 3, 4, 5], cv=5)

    # ── 5. Ridge-regularized polynomial ───────────────────────────────────
    print("\n" + "=" * 65)
    print("5️⃣  Ridge-Regularized Polynomial (degree=4)")
    print("=" * 65)
    ridge_df = train_ridge_poly(
        X_train, X_test, y_train, y_test,
        degree=4,
        alphas=[0.01, 0.1, 1.0, 10.0, 100.0]
    )

    print("\n✅ All Polynomial Regression techniques demonstrated successfully!")
