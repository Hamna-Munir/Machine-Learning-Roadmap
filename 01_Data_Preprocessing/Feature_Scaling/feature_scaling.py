# =============================================================================
# 📦 Feature Scaling — Reusable ML Script
# =============================================================================
# Author   : Hamna Munir
# Topic    : 01_Data_Preprocessing / Feature_Scaling
# File     : feature_scaling.py
# =============================================================================

# -----------------------------------------------------------------------------
# 📚 Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    MaxAbsScaler,
    Normalizer,
    PowerTransformer,
    FunctionTransformer,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 🔧 1. MIN-MAX SCALING (Normalization)
# =============================================================================

def apply_minmax_scaling(X_train: pd.DataFrame,
                          X_test: pd.DataFrame,
                          columns: list,
                          feature_range: tuple = (0, 1)) -> tuple:
    """
    Applies Min-Max Scaling to specified columns.

    Formula : X_scaled = (X - X_min) / (X_max - X_min)
    Range   : [0, 1] by default (or custom feature_range)

    Best for:
        - Neural Networks
        - Data without extreme outliers
        - Algorithms needing bounded input

    Args:
        X_train       : Training features DataFrame
        X_test        : Test features DataFrame
        columns       : List of numeric columns to scale
        feature_range : Output range tuple (default: (0, 1))

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    X_train, X_test = X_train.copy(), X_test.copy()
    scaler = MinMaxScaler(feature_range=feature_range)

    X_train[columns] = scaler.fit_transform(X_train[columns])
    X_test[columns]  = scaler.transform(X_test[columns])

    print(f"[MinMaxScaler] Range: {feature_range} | Columns: {columns}")
    return X_train, X_test, scaler


# =============================================================================
# 🔧 2. STANDARDIZATION (Z-Score Scaling)
# =============================================================================

def apply_standard_scaling(X_train: pd.DataFrame,
                             X_test: pd.DataFrame,
                             columns: list) -> tuple:
    """
    Applies Z-Score Standardization to specified columns.

    Formula : X_scaled = (X - mean) / std
    Output  : mean=0, std=1

    Best for:
        - Linear/Logistic Regression, SVM
        - PCA, LDA
        - Gradient-descent-based algorithms
        - General-purpose default scaler

    Args:
        X_train : Training features DataFrame
        X_test  : Test features DataFrame
        columns : List of numeric columns to scale

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    X_train, X_test = X_train.copy(), X_test.copy()
    scaler = StandardScaler()

    X_train[columns] = scaler.fit_transform(X_train[columns])
    X_test[columns]  = scaler.transform(X_test[columns])

    print(f"[StandardScaler] mean={np.round(scaler.mean_, 2)} | std={np.round(scaler.scale_, 2)}")
    return X_train, X_test, scaler


# =============================================================================
# 🔧 3. ROBUST SCALING
# =============================================================================

def apply_robust_scaling(X_train: pd.DataFrame,
                          X_test: pd.DataFrame,
                          columns: list) -> tuple:
    """
    Applies Robust Scaling using median and IQR.

    Formula : X_scaled = (X - median) / IQR
    Output  : Centered at 0, unbounded

    Best for:
        - Data with significant outliers
        - Financial, sensor, or medical data

    Args:
        X_train : Training features DataFrame
        X_test  : Test features DataFrame
        columns : List of numeric columns to scale

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    X_train, X_test = X_train.copy(), X_test.copy()
    scaler = RobustScaler()

    X_train[columns] = scaler.fit_transform(X_train[columns])
    X_test[columns]  = scaler.transform(X_test[columns])

    print(f"[RobustScaler] center={np.round(scaler.center_, 2)} | scale={np.round(scaler.scale_, 2)}")
    return X_train, X_test, scaler


# =============================================================================
# 🔧 4. MAXABS SCALING
# =============================================================================

def apply_maxabs_scaling(X_train: pd.DataFrame,
                          X_test: pd.DataFrame,
                          columns: list) -> tuple:
    """
    Applies MaxAbs Scaling — divides by the maximum absolute value.

    Formula : X_scaled = X / |X_max|
    Range   : [-1, 1]

    Best for:
        - Sparse data (zeros remain zero)
        - TF-IDF or count matrices

    Args:
        X_train : Training features DataFrame
        X_test  : Test features DataFrame
        columns : List of numeric columns to scale

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, fitted_scaler)
    """
    X_train, X_test = X_train.copy(), X_test.copy()
    scaler = MaxAbsScaler()

    X_train[columns] = scaler.fit_transform(X_train[columns])
    X_test[columns]  = scaler.transform(X_test[columns])

    print(f"[MaxAbsScaler] max_abs={np.round(scaler.max_abs_, 2)}")
    return X_train, X_test, scaler


# =============================================================================
# 🔧 5. LOG TRANSFORMATION
# =============================================================================

def apply_log_transform(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Applies log(X + 1) transformation to reduce right skewness.

    Formula : X_scaled = log(X + 1)

    Best for:
        - Right-skewed features (income, prices, counts)
        - Positive-valued features only

    Args:
        df      : Input DataFrame
        columns : List of numeric columns to transform

    Returns:
        DataFrame with log-transformed columns
    """
    df = df.copy()
    for col in columns:
        if (df[col] < 0).any():
            print(f"[LogTransform] ⚠️  '{col}' has negative values — skipping")
            continue
        skew_before = round(df[col].skew(), 4)
        df[col] = np.log1p(df[col])
        skew_after  = round(df[col].skew(), 4)
        print(f"[LogTransform] '{col}' | skew: {skew_before} → {skew_after}")

    return df


# =============================================================================
# 🔧 6. POWER TRANSFORMATION (Yeo-Johnson / Box-Cox)
# =============================================================================

def apply_power_transform(X_train: pd.DataFrame,
                           X_test: pd.DataFrame,
                           columns: list,
                           method: str = "yeo-johnson") -> tuple:
    """
    Applies Power Transformation to make distributions more Gaussian.

    Methods:
        - 'yeo-johnson' : Works with zero and negative values (recommended)
        - 'box-cox'     : Positive values only

    Best for:
        - Heavily skewed continuous features
        - When normality assumption is required (linear models)

    Args:
        X_train : Training features DataFrame
        X_test  : Test features DataFrame
        columns : List of numeric columns to transform
        method  : 'yeo-johnson' (default) or 'box-cox'

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, fitted_transformer)
    """
    X_train, X_test = X_train.copy(), X_test.copy()
    transformer = PowerTransformer(method=method, standardize=True)

    X_train[columns] = transformer.fit_transform(X_train[columns])
    X_test[columns]  = transformer.transform(X_test[columns])

    print(f"[PowerTransformer] method='{method}' | lambdas={np.round(transformer.lambdas_, 4)}")
    return X_train, X_test, transformer


# =============================================================================
# 🔧 7. UNIT VECTOR SCALING (Normalizer — per row)
# =============================================================================

def apply_normalizer(X_train: pd.DataFrame,
                      X_test: pd.DataFrame,
                      columns: list,
                      norm: str = "l2") -> tuple:
    """
    Scales each sample (row) to unit norm.

    Formula : X_scaled = X / ||X||  (L2 norm by default)

    Note    : Applied per ROW (not per column — unlike all other scalers)

    Best for:
        - Text classification (TF-IDF vectors)
        - Cosine similarity computations

    Args:
        X_train : Training features DataFrame
        X_test  : Test features DataFrame
        columns : List of numeric columns to normalize
        norm    : Norm type — 'l1', 'l2' (default), or 'max'

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, fitted_normalizer)
    """
    X_train, X_test = X_train.copy(), X_test.copy()
    normalizer = Normalizer(norm=norm)

    X_train[columns] = normalizer.fit_transform(X_train[columns])
    X_test[columns]  = normalizer.transform(X_test[columns])

    print(f"[Normalizer] norm='{norm}' | Applied per-row on: {columns}")
    return X_train, X_test, normalizer


# =============================================================================
# 🔧 8. UTILITY — IDENTIFY NUMERIC COLUMNS
# =============================================================================

def get_numeric_columns(df: pd.DataFrame,
                         exclude: list = None) -> list:
    """
    Returns a list of numeric columns in the DataFrame.

    Args:
        df      : Input DataFrame
        exclude : Optional list of columns to exclude (e.g., target)

    Returns:
        List of numeric column names
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if exclude:
        num_cols = [c for c in num_cols if c not in exclude]
    print(f"[Numeric Columns] {num_cols}")
    return num_cols


# =============================================================================
# 🔧 9. UTILITY — SCALING REPORT (before vs after)
# =============================================================================

def scaling_report(df_before: pd.DataFrame,
                    df_after: pd.DataFrame,
                    columns: list) -> pd.DataFrame:
    """
    Compares feature statistics before and after scaling.

    Args:
        df_before : Original DataFrame
        df_after  : Scaled DataFrame
        columns   : Columns to compare

    Returns:
        DataFrame with before/after stats
    """
    rows = []
    for col in columns:
        rows.append({
            "Column"    : col,
            "Mean (before)" : round(df_before[col].mean(), 4),
            "Mean (after)"  : round(df_after[col].mean(), 4),
            "Std (before)"  : round(df_before[col].std(), 4),
            "Std (after)"   : round(df_after[col].std(), 4),
            "Min (before)"  : round(df_before[col].min(), 4),
            "Min (after)"   : round(df_after[col].min(), 4),
            "Max (before)"  : round(df_before[col].max(), 4),
            "Max (after)"   : round(df_after[col].max(), 4),
        })

    report = pd.DataFrame(rows)
    print(report.to_string(index=False))
    return report


# =============================================================================
# 🔧 10. SKLEARN PIPELINE EXAMPLE
# =============================================================================

def build_scaling_pipeline(scaler_name: str = "standard") -> Pipeline:
    """
    Builds a reusable sklearn Pipeline with a scaler.

    Args:
        scaler_name : One of 'standard', 'minmax', 'robust', 'maxabs'

    Returns:
        sklearn Pipeline object with the chosen scaler

    Example usage:
        pipe = build_scaling_pipeline('robust')
        pipe.fit(X_train)
        X_scaled = pipe.transform(X_test)
    """
    scalers = {
        "standard" : StandardScaler(),
        "minmax"   : MinMaxScaler(),
        "robust"   : RobustScaler(),
        "maxabs"   : MaxAbsScaler(),
    }

    if scaler_name not in scalers:
        raise ValueError(f"Invalid scaler. Choose from: {list(scalers.keys())}")

    pipeline = Pipeline([("scaler", scalers[scaler_name])])
    print(f"[Pipeline] Built with scaler: '{scaler_name}'")
    return pipeline


# =============================================================================
# 🚀 MAIN — Demo with Synthetic Data
# =============================================================================

if __name__ == "__main__":

    # ── Sample Dataset ──────────────────────────────────────────────────────
    np.random.seed(42)
    data = {
        "Age"    : np.random.randint(20, 60, 100),
        "Salary" : np.random.randint(30_000, 200_000, 100),
        "Score"  : np.random.uniform(0, 100, 100),
        "Income" : np.random.exponential(scale=50_000, size=100),  # skewed
        "Target" : np.random.randint(0, 2, 100),
    }
    df = pd.DataFrame(data)

    print("=" * 60)
    print("📊 Original Dataset — First 5 Rows")
    print("=" * 60)
    print(df.head())

    # ── Train-Test Split ──────────────────────────────────────────────────
    feature_cols = ["Age", "Salary", "Score", "Income"]
    X = df[feature_cols]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── 1. Min-Max Scaling ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("1️⃣  Min-Max Scaling")
    print("=" * 60)
    X_tr_mm, X_te_mm, _ = apply_minmax_scaling(X_train, X_test, feature_cols)
    scaling_report(X_train, X_tr_mm, feature_cols)

    # ── 2. Standardization ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2️⃣  Standardization (Z-Score)")
    print("=" * 60)
    X_tr_std, X_te_std, _ = apply_standard_scaling(X_train, X_test, feature_cols)
    scaling_report(X_train, X_tr_std, feature_cols)

    # ── 3. Robust Scaling ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3️⃣  Robust Scaling")
    print("=" * 60)
    X_tr_rob, X_te_rob, _ = apply_robust_scaling(X_train, X_test, feature_cols)
    scaling_report(X_train, X_tr_rob, feature_cols)

    # ── 4. Log Transformation ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("5️⃣  Log Transformation (skewed feature)")
    print("=" * 60)
    df_log = apply_log_transform(df.copy(), columns=["Income"])

    # ── 5. Power Transformation ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("6️⃣  Power Transformation (Yeo-Johnson)")
    print("=" * 60)
    X_tr_pow, X_te_pow, _ = apply_power_transform(
        X_train, X_test, columns=["Income"], method="yeo-johnson"
    )

    # ── 6. Pipeline Example ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🔧  Sklearn Pipeline")
    print("=" * 60)
    pipe = build_scaling_pipeline("robust")
    pipe.fit(X_train)
    X_pipe_scaled = pipe.transform(X_test)
    print(f"Pipeline output shape: {X_pipe_scaled.shape}")

    print("\n✅ All scaling techniques demonstrated successfully!")
