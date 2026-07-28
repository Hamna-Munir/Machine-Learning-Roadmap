# =============================================================================
# 📦 Linear Regression — Reusable ML Script
# =============================================================================
# Author   : Hamna Munir
# Topic    : 03_Supervised_Learning / Regression / Linear_Regression
# File     : linear_regression.py
# =============================================================================

# -----------------------------------------------------------------------------
# 📚 Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    r2_score, mean_absolute_percentage_error
)
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
from scipy import stats

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 🔧 1. TRAIN SIMPLE LINEAR REGRESSION (OLS)
# =============================================================================

def train_linear_regression(X_train: pd.DataFrame,
                              X_test: pd.DataFrame,
                              y_train: pd.Series,
                              y_test: pd.Series,
                              fit_intercept: bool = True) -> dict:
    """
    Trains a Linear Regression model using sklearn's OLS implementation.

    Args:
        X_train       : Training features
        X_test        : Test features
        y_train       : Training target
        y_test        : Test target
        fit_intercept : Whether to fit an intercept term (default: True)

    Returns:
        Dictionary with model, predictions, and evaluation metrics
    """
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    metrics = evaluate_regression(y_train, y_pred_train, y_test, y_pred_test)

    print(f"[LinearRegression] Intercept: {model.intercept_:.4f}")
    print(f"  Coefficients: {dict(zip(X_train.columns, model.coef_.round(4)))}")
    _print_metrics(metrics)

    return {
        "model"          : model,
        "y_pred_train"   : y_pred_train,
        "y_pred_test"    : y_pred_test,
        "metrics"        : metrics,
        "coefficients"   : pd.Series(model.coef_, index=X_train.columns),
        "intercept"      : model.intercept_,
    }


# =============================================================================
# 🔧 2. TRAIN WITH GRADIENT DESCENT (SGDRegressor)
# =============================================================================

def train_sgd_regression(X_train: pd.DataFrame,
                           X_test: pd.DataFrame,
                           y_train: pd.Series,
                           y_test: pd.Series,
                           learning_rate: str = "invscaling",
                           eta0: float = 0.01,
                           max_iter: int = 1000,
                           random_state: int = 42) -> dict:
    """
    Trains Linear Regression via Stochastic Gradient Descent (SGD).

    Best for:
        - Very large datasets where matrix inversion is too slow
        - Online learning / streaming data

    Args:
        X_train       : Training features (should be scaled)
        X_test        : Test features
        y_train       : Training target
        y_test        : Test target
        learning_rate : Learning rate schedule ('constant', 'invscaling', 'adaptive')
        eta0          : Initial learning rate (default: 0.01)
        max_iter      : Maximum number of passes over training data
        random_state  : Reproducibility seed

    Returns:
        Dictionary with model, predictions, and evaluation metrics
    """
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = SGDRegressor(
        loss="squared_error",
        learning_rate=learning_rate,
        eta0=eta0,
        max_iter=max_iter,
        random_state=random_state,
        tol=1e-4,
    )
    model.fit(X_train_sc, y_train)

    y_pred_train = model.predict(X_train_sc)
    y_pred_test  = model.predict(X_test_sc)

    metrics = evaluate_regression(y_train, y_pred_train, y_test, y_pred_test)

    print(f"[SGDRegressor] eta0={eta0} | max_iter={max_iter}")
    _print_metrics(metrics)

    return {
        "model"       : model,
        "scaler"      : scaler,
        "y_pred_train": y_pred_train,
        "y_pred_test" : y_pred_test,
        "metrics"     : metrics,
    }


# =============================================================================
# 🔧 3. STATSMODELS OLS — STATISTICAL SUMMARY
# =============================================================================

def train_ols_statsmodels(X_train: pd.DataFrame,
                            y_train: pd.Series) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fits OLS using statsmodels — provides full statistical summary including
    p-values, confidence intervals, F-statistic, and assumption tests.

    Best for:
        - Statistical inference (are coefficients significant?)
        - Checking assumptions formally

    Args:
        X_train : Training features DataFrame
        y_train : Training target Series

    Returns:
        Fitted statsmodels OLS result object
    """
    X_const = sm.add_constant(X_train)   # add intercept column
    model   = sm.OLS(y_train, X_const).fit()

    print("[Statsmodels OLS] Full Summary:")
    print(model.summary())

    return model


# =============================================================================
# 🔧 4. CROSS-VALIDATION
# =============================================================================

def cross_validate_linear_regression(X: pd.DataFrame,
                                       y: pd.Series,
                                       cv: int = 5,
                                       scoring: str = "r2") -> dict:
    """
    Performs K-Fold Cross-Validation on Linear Regression.

    Args:
        X       : Full feature DataFrame
        y       : Full target Series
        cv      : Number of folds (default: 5)
        scoring : Scoring metric — 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'

    Returns:
        Dictionary with fold scores, mean, and std
    """
    model  = LinearRegression()
    kf     = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)

    result = {
        "scores"      : scores,
        "mean"        : scores.mean(),
        "std"         : scores.std(),
        "fold_results": {f"Fold {i+1}": round(s, 4) for i, s in enumerate(scores)},
    }

    print(f"[Cross-Validation] K={cv} | {scoring.upper()}: "
          f"{scores.mean():.4f} ± {scores.std():.4f}")
    for fold, score in result["fold_results"].items():
        print(f"  {fold}: {score}")

    return result


# =============================================================================
# 🔧 5. SKLEARN PIPELINE (SCALE + FIT)
# =============================================================================

def build_linear_pipeline(scale: bool = True) -> Pipeline:
    """
    Builds a reusable sklearn Pipeline: StandardScaler + LinearRegression.

    Args:
        scale : Whether to include StandardScaler step (default: True)

    Returns:
        sklearn Pipeline object

    Usage:
        pipe = build_linear_pipeline()
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
    """
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", LinearRegression()))

    pipeline = Pipeline(steps)
    print(f"[Pipeline] Built: {' → '.join([s for s, _ in steps])}")
    return pipeline


# =============================================================================
# 🔧 6. EVALUATE REGRESSION MODEL
# =============================================================================

def evaluate_regression(y_train: pd.Series,
                          y_pred_train: np.ndarray,
                          y_test: pd.Series,
                          y_pred_test: np.ndarray) -> dict:
    """
    Computes a comprehensive set of regression evaluation metrics
    for both training and test sets.

    Metrics:
        MAE   — Mean Absolute Error
        MSE   — Mean Squared Error
        RMSE  — Root Mean Squared Error
        MAPE  — Mean Absolute Percentage Error
        R²    — Coefficient of Determination
        Adj R²— Adjusted R² (requires n_features from X shape)

    Args:
        y_train      : True training labels
        y_pred_train : Predicted training values
        y_test       : True test labels
        y_pred_test  : Predicted test values

    Returns:
        Dictionary with all metrics for train and test sets
    """
    def metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return {
            "MAE"  : round(mean_absolute_error(y_true, y_pred), 4),
            "MSE"  : round(mse, 4),
            "RMSE" : round(np.sqrt(mse), 4),
            "MAPE" : round(mean_absolute_percentage_error(y_true, y_pred) * 100, 4),
            "R²"   : round(r2_score(y_true, y_pred), 4),
        }

    return {
        "train": metrics(y_train, y_pred_train),
        "test" : metrics(y_test,  y_pred_test),
    }


# =============================================================================
# 🔧 7. RESIDUAL ANALYSIS
# =============================================================================

def residual_analysis(y_test: pd.Series,
                       y_pred: np.ndarray) -> dict:
    """
    Computes residual statistics and runs the Shapiro-Wilk normality test.

    Residuals = y_true − y_predicted

    Checks:
        - Mean of residuals (should be ≈ 0)
        - Std of residuals
        - Shapiro-Wilk test (residuals should be normally distributed)
        - Skewness and Kurtosis of residuals

    Args:
        y_test  : True target values
        y_pred  : Predicted target values

    Returns:
        Dictionary with residual statistics and test results
    """
    residuals = np.array(y_test) - np.array(y_pred)
    sw_stat, sw_p = stats.shapiro(residuals[:min(5000, len(residuals))])

    result = {
        "residuals"        : residuals,
        "mean"             : round(residuals.mean(), 6),
        "std"              : round(residuals.std(), 4),
        "skewness"         : round(stats.skew(residuals), 4),
        "kurtosis"         : round(stats.kurtosis(residuals), 4),
        "shapiro_stat"     : round(sw_stat, 4),
        "shapiro_p"        : round(sw_p, 4),
        "normality_ok"     : sw_p > 0.05,
    }

    print(f"\n[Residual Analysis]")
    print(f"  Mean        : {result['mean']:.6f}  (should be ≈ 0)")
    print(f"  Std Dev     : {result['std']:.4f}")
    print(f"  Skewness    : {result['skewness']:.4f}")
    print(f"  Kurtosis    : {result['kurtosis']:.4f}")
    print(f"  Shapiro-Wilk: W={result['shapiro_stat']:.4f}  p={result['shapiro_p']:.4f}  "
          f"{'✅ Normal' if result['normality_ok'] else '❌ Non-Normal'}")

    return result


# =============================================================================
# 🔧 8. VIF (VARIANCE INFLATION FACTOR)
# =============================================================================

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Variance Inflation Factor (VIF) for all features.

    VIF measures how much variance of a coefficient is inflated
    due to multicollinearity with other features.

    Interpretation:
        VIF = 1       → No multicollinearity
        VIF = 1–5     → Low (acceptable)
        VIF = 5–10    → Moderate (investigate)
        VIF > 10      → High (consider dropping)

    Args:
        X : Feature DataFrame

    Returns:
        DataFrame with VIF scores sorted descending
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X_sc = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    vif_df = pd.DataFrame({
        "Feature" : X.columns,
        "VIF"     : [variance_inflation_factor(X_sc.values, i)
                     for i in range(X_sc.shape[1])],
    }).sort_values("VIF", ascending=False).reset_index(drop=True)

    vif_df["Status"] = vif_df["VIF"].apply(
        lambda v: "🔴 High"     if v > 10 else
                  ("🟡 Moderate" if v > 5  else "🟢 OK")
    )

    print("\n[VIF Analysis]")
    print(vif_df.round(2).to_string(index=False))
    return vif_df


# =============================================================================
# 🔧 9. COEFFICIENT SUMMARY TABLE
# =============================================================================

def coefficient_summary(model: LinearRegression,
                          feature_names: list,
                          X_train: pd.DataFrame,
                          y_train: pd.Series) -> pd.DataFrame:
    """
    Builds a comprehensive coefficient summary table using statsmodels
    for p-values and confidence intervals.

    Args:
        model        : Fitted sklearn LinearRegression model
        feature_names: List of feature column names
        X_train      : Training features (for statsmodels refit)
        y_train      : Training target

    Returns:
        DataFrame with coefficients, p-values, and confidence intervals
    """
    X_const = sm.add_constant(X_train)
    sm_model = sm.OLS(y_train, X_const).fit()

    summary = pd.DataFrame({
        "Feature"   : ["Intercept"] + list(feature_names),
        "Coef"      : sm_model.params.round(4).values,
        "Std Error" : sm_model.bse.round(4).values,
        "t-Stat"    : sm_model.tvalues.round(4).values,
        "p-value"   : sm_model.pvalues.round(4).values,
        "CI Lower"  : sm_model.conf_int()[0].round(4).values,
        "CI Upper"  : sm_model.conf_int()[1].round(4).values,
        "Significant": ["✅" if p < 0.05 else "❌" for p in sm_model.pvalues],
    })

    print("\n[Coefficient Summary]")
    print(summary.to_string(index=False))
    return summary


# =============================================================================
# 🔧 10. UTILITY — REGRESSION REPORT
# =============================================================================

def regression_report(y_test: pd.Series,
                        y_pred: np.ndarray,
                        model_name: str = "Linear Regression") -> pd.DataFrame:
    """
    Prints and returns a formatted regression evaluation report.

    Args:
        y_test     : True target values
        y_pred     : Predicted target values
        model_name : Name of the model (for display)

    Returns:
        DataFrame with all metrics
    """
    mse = mean_squared_error(y_test, y_pred)

    report = pd.DataFrame([{
        "Model"     : model_name,
        "MAE"       : round(mean_absolute_error(y_test, y_pred), 4),
        "MSE"       : round(mse, 4),
        "RMSE"      : round(np.sqrt(mse), 4),
        "MAPE %"    : round(mean_absolute_percentage_error(y_test, y_pred) * 100, 4),
        "R²"        : round(r2_score(y_test, y_pred), 4),
    }])

    print(f"\n📊 Regression Report — {model_name}")
    print(report.to_string(index=False))
    return report


# =============================================================================
# 🔧 HELPER — PRINT METRICS
# =============================================================================

def _print_metrics(metrics: dict) -> None:
    """Internal helper to print train/test evaluation metrics."""
    for split in ["train", "test"]:
        m = metrics[split]
        print(f"  [{split.upper():5s}] MAE={m['MAE']:>10.4f} | RMSE={m['RMSE']:>10.4f} | "
              f"R²={m['R²']:>7.4f} | MAPE={m['MAPE']:>7.2f}%")


# =============================================================================
# 🚀 MAIN — Demo with Synthetic Dataset
# =============================================================================

if __name__ == "__main__":

    # ── Synthetic Dataset ──────────────────────────────────────────────────
    np.random.seed(42)
    n = 500

    X = pd.DataFrame({
        "Experience" : np.random.randint(0, 30, n),
        "Score"      : np.random.uniform(40, 100, n),
        "Age"        : np.random.randint(22, 60, n),
    })

    # Target: Salary with a linear relationship + noise
    y = pd.Series(
        20_000
        + 2_500  * X["Experience"]
        + 500    * X["Score"]
        + 300    * X["Age"]
        + np.random.normal(0, 5_000, n),
        name="Salary"
    )

    print("=" * 65)
    print("📊 Dataset Info")
    print("=" * 65)
    print(f"Shape  : {X.shape}")
    print(f"Target : {y.name}  |  Mean: £{y.mean():,.0f}  |  Std: £{y.std():,.0f}")

    # ── Train-Test Split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── 1. OLS Linear Regression ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("1️⃣  OLS Linear Regression (sklearn)")
    print("=" * 65)
    result = train_linear_regression(X_train, X_test, y_train, y_test)

    # ── 2. Statsmodels OLS ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("2️⃣  Statsmodels OLS — Statistical Summary")
    print("=" * 65)
    sm_model = train_ols_statsmodels(X_train, y_train)

    # ── 3. Coefficient Summary ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("3️⃣  Coefficient Summary with p-values")
    print("=" * 65)
    coef_df = coefficient_summary(
        result["model"], X_train.columns, X_train, y_train
    )

    # ── 4. VIF Analysis ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("4️⃣  VIF Multicollinearity Analysis")
    print("=" * 65)
    vif_df = compute_vif(X_train)

    # ── 5. Residual Analysis ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("5️⃣  Residual Analysis")
    print("=" * 65)
    res = residual_analysis(y_test, result["y_pred_test"])

    # ── 6. Cross-Validation ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("6️⃣  5-Fold Cross-Validation")
    print("=" * 65)
    cv_result = cross_validate_linear_regression(X, y, cv=5, scoring="r2")

    # ── 7. Pipeline ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("7️⃣  Sklearn Pipeline (Scale + LinearRegression)")
    print("=" * 65)
    pipe = build_linear_pipeline(scale=True)
    pipe.fit(X_train, y_train)
    y_pipe_pred = pipe.predict(X_test)
    regression_report(y_test, y_pipe_pred, "Pipeline (Scaled LinearRegression)")

    # ── 8. SGD Regression ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("8️⃣  SGD Regression (Gradient Descent)")
    print("=" * 65)
    sgd_result = train_sgd_regression(X_train, X_test, y_train, y_test)

    print("\n✅ All Linear Regression techniques demonstrated successfully!")
