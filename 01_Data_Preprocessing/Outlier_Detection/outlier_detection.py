# =============================================================================
# 📦 Outlier Detection — Reusable ML Script
# =============================================================================
# Author   : Hamna Munir
# Topic    : 01_Data_Preprocessing / Outlier_Detection
# File     : outlier_detection.py
# =============================================================================

# -----------------------------------------------------------------------------
# 📚 Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 🔧 1. Z-SCORE METHOD
# =============================================================================

def detect_zscore(df: pd.DataFrame,
                   columns: list,
                   threshold: float = 3.0) -> pd.DataFrame:
    """
    Detects outliers using the Z-Score method.

    Formula : Z = (X - μ) / σ
    Rule    : |Z| > threshold → outlier

    Best for:
        - Normally distributed, univariate features
        - Quick initial screening

    Args:
        df        : Input DataFrame
        columns   : List of numeric columns to check
        threshold : Z-score cutoff (default: 3.0)

    Returns:
        DataFrame with boolean outlier flags (True = outlier)
        Columns: '{col}_zscore_outlier'
    """
    df = df.copy()
    for col in columns:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        df[f"{col}_zscore_outlier"] = np.abs(stats.zscore(df[col])) > threshold
        count = df[f"{col}_zscore_outlier"].sum()
        print(f"[Z-Score] '{col}' → {count} outliers detected (threshold={threshold})")
    return df


# =============================================================================
# 🔧 2. IQR METHOD
# =============================================================================

def detect_iqr(df: pd.DataFrame,
                columns: list,
                factor: float = 1.5) -> pd.DataFrame:
    """
    Detects outliers using the IQR (Interquartile Range) method.

    Formula:
        IQR         = Q3 - Q1
        Lower Fence = Q1 - factor × IQR
        Upper Fence = Q3 + factor × IQR

    Best for:
        - Skewed or non-normal distributions
        - Robust univariate outlier detection

    Args:
        df      : Input DataFrame
        columns : List of numeric columns to check
        factor  : IQR multiplier (default: 1.5; use 3.0 for extreme outliers)

    Returns:
        DataFrame with boolean outlier flags and fence values printed
        Columns: '{col}_iqr_outlier'
    """
    df = df.copy()
    for col in columns:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        df[f"{col}_iqr_outlier"] = (df[col] < lower) | (df[col] > upper)
        count = df[f"{col}_iqr_outlier"].sum()
        print(f"[IQR] '{col}' → Q1={Q1:.2f}, Q3={Q3:.2f}, "
              f"IQR={IQR:.2f} | Fence: [{lower:.2f}, {upper:.2f}] | {count} outliers")
    return df


def get_iqr_bounds(series: pd.Series, factor: float = 1.5) -> tuple:
    """
    Returns the lower and upper IQR fence values for a Series.

    Args:
        series : Numeric pandas Series
        factor : IQR multiplier (default: 1.5)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    Q1  = series.quantile(0.25)
    Q3  = series.quantile(0.75)
    IQR = Q3 - Q1
    return Q1 - factor * IQR, Q3 + factor * IQR


# =============================================================================
# 🔧 3. MODIFIED Z-SCORE (MAD)
# =============================================================================

def detect_modified_zscore(df: pd.DataFrame,
                             columns: list,
                             threshold: float = 3.5) -> pd.DataFrame:
    """
    Detects outliers using the Modified Z-Score (MAD — Median Absolute Deviation).

    Formula:
        MAD      = median(|X - median(X)|)
        ModZ     = 0.6745 × (X - median(X)) / MAD
        Outlier  : |ModZ| > threshold

    Best for:
        - Small datasets
        - When Z-Score is too sensitive to its own outliers
        - Robust alternative to standard Z-Score

    Args:
        df        : Input DataFrame
        columns   : List of numeric columns to check
        threshold : Modified Z-score cutoff (default: 3.5)

    Returns:
        DataFrame with boolean outlier flags
        Columns: '{col}_mad_outlier'
    """
    df = df.copy()
    for col in columns:
        median  = df[col].median()
        mad     = np.median(np.abs(df[col] - median))
        mod_z   = 0.6745 * (df[col] - median) / (mad + 1e-10)
        df[f"{col}_mad_outlier"] = np.abs(mod_z) > threshold
        count = df[f"{col}_mad_outlier"].sum()
        print(f"[MAD] '{col}' → median={median:.2f}, MAD={mad:.2f} | {count} outliers")
    return df


# =============================================================================
# 🔧 4. ISOLATION FOREST
# =============================================================================

def detect_isolation_forest(X: pd.DataFrame,
                              contamination: float = 0.05,
                              n_estimators: int = 100,
                              random_state: int = 42) -> pd.Series:
    """
    Detects multivariate outliers using Isolation Forest.

    Core idea:
        Anomalies are easier to isolate — they need fewer random splits.
        Returns -1 for outliers, 1 for inliers.

    Best for:
        - High-dimensional data
        - No distributional assumptions
        - Large datasets

    Args:
        X             : Feature DataFrame (numeric only)
        contamination : Expected proportion of outliers (default: 0.05)
        n_estimators  : Number of trees (default: 100)
        random_state  : Reproducibility seed (default: 42)

    Returns:
        pandas Series with labels: -1 = outlier, 1 = inlier
    """
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state
    )
    labels = model.fit_predict(X)
    scores = model.decision_function(X)

    n_outliers = (labels == -1).sum()
    print(f"[IsolationForest] contamination={contamination} | "
          f"{n_outliers} outliers detected out of {len(X)}")

    return pd.Series(labels, index=X.index, name="if_label"), \
           pd.Series(scores, index=X.index, name="if_score")


# =============================================================================
# 🔧 5. LOCAL OUTLIER FACTOR (LOF)
# =============================================================================

def detect_lof(X: pd.DataFrame,
                n_neighbors: int = 20,
                contamination: float = 0.05) -> pd.Series:
    """
    Detects outliers using Local Outlier Factor (LOF).

    Core idea:
        Compares local density of a point to its neighbors.
        LOF >> 1 → point is in a much sparser region than neighbors → outlier.

    Best for:
        - Datasets with varying density clusters
        - When global thresholds miss local anomalies

    Args:
        X             : Feature DataFrame (numeric only)
        n_neighbors   : Number of neighbors (default: 20)
        contamination : Expected proportion of outliers (default: 0.05)

    Returns:
        Tuple of (labels Series, scores Series)
        labels: -1 = outlier, 1 = inlier
    """
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination
    )
    labels = model.fit_predict(X)
    scores = model.negative_outlier_factor_

    n_outliers = (labels == -1).sum()
    print(f"[LOF] n_neighbors={n_neighbors} | "
          f"{n_outliers} outliers detected out of {len(X)}")

    return pd.Series(labels, index=X.index, name="lof_label"), \
           pd.Series(scores, index=X.index, name="lof_score")


# =============================================================================
# 🔧 6. DBSCAN
# =============================================================================

def detect_dbscan(X: pd.DataFrame,
                   eps: float = 0.5,
                   min_samples: int = 5,
                   scale: bool = True) -> pd.Series:
    """
    Detects outliers using DBSCAN clustering.

    Core idea:
        Points that do not belong to any dense cluster are labeled as
        noise (cluster = -1), which are treated as outliers.

    Best for:
        - Spatial data
        - Arbitrary cluster shapes
        - When outliers = noise points

    Args:
        X           : Feature DataFrame (numeric only)
        eps         : Maximum distance between neighbors (default: 0.5)
        min_samples : Minimum cluster size (default: 5)
        scale       : Standardize features before DBSCAN (recommended: True)

    Returns:
        pandas Series with cluster labels (-1 = noise/outlier)
    """
    X_scaled = StandardScaler().fit_transform(X) if scale else X.values
    model  = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)

    n_outliers = (labels == -1).sum()
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[DBSCAN] eps={eps}, min_samples={min_samples} | "
          f"{n_clusters} clusters | {n_outliers} noise/outlier points")

    return pd.Series(labels, index=X.index, name="dbscan_label")


# =============================================================================
# 🔧 7. ELLIPTIC ENVELOPE
# =============================================================================

def detect_elliptic_envelope(X: pd.DataFrame,
                               contamination: float = 0.05,
                               random_state: int = 42) -> pd.Series:
    """
    Detects multivariate outliers using Elliptic Envelope (Robust Covariance).

    Core idea:
        Assumes data follows a Gaussian distribution.
        Fits a robust covariance ellipse and flags points outside as outliers.
        Uses Mahalanobis distance.

    Best for:
        - Normally distributed multivariate data
        - Low-dimensional feature spaces

    Args:
        X             : Feature DataFrame (numeric only)
        contamination : Expected proportion of outliers (default: 0.05)
        random_state  : Reproducibility seed (default: 42)

    Returns:
        pandas Series with labels: -1 = outlier, 1 = inlier
    """
    model  = EllipticEnvelope(contamination=contamination, random_state=random_state)
    labels = model.fit_predict(X)

    n_outliers = (labels == -1).sum()
    print(f"[EllipticEnvelope] contamination={contamination} | "
          f"{n_outliers} outliers detected out of {len(X)}")

    return pd.Series(labels, index=X.index, name="ee_label")


# =============================================================================
# 🔧 8. WINSORIZATION (CAPPING)
# =============================================================================

def apply_winsorization(df: pd.DataFrame,
                          columns: list,
                          lower_pct: float = 0.05,
                          upper_pct: float = 0.95) -> pd.DataFrame:
    """
    Caps outliers at specified percentile boundaries (Winsorization).

    Formula:
        Values < lower_pct percentile → replaced with lower_pct percentile
        Values > upper_pct percentile → replaced with upper_pct percentile

    Best for:
        - Retaining all rows while reducing outlier influence
        - Regression tasks where row count matters

    Args:
        df        : Input DataFrame
        columns   : List of numeric columns to winsorize
        lower_pct : Lower percentile cap (default: 0.05 = 5th percentile)
        upper_pct : Upper percentile cap (default: 0.95 = 95th percentile)

    Returns:
        DataFrame with capped values
    """
    df = df.copy()
    for col in columns:
        lower = df[col].quantile(lower_pct)
        upper = df[col].quantile(upper_pct)
        before_range = f"[{df[col].min():.2f}, {df[col].max():.2f}]"
        df[col] = df[col].clip(lower=lower, upper=upper)
        after_range  = f"[{df[col].min():.2f}, {df[col].max():.2f}]"
        print(f"[Winsorize] '{col}' | Before: {before_range} → After: {after_range}")
    return df


# =============================================================================
# 🔧 9. UTILITY — REMOVE DETECTED OUTLIERS
# =============================================================================

def remove_outliers(df: pd.DataFrame, outlier_mask: pd.Series) -> pd.DataFrame:
    """
    Removes rows flagged as outliers from the DataFrame.

    Args:
        df            : Original DataFrame
        outlier_mask  : Boolean Series (True = outlier to remove)

    Returns:
        Cleaned DataFrame with outlier rows dropped
    """
    original_len = len(df)
    df_clean = df[~outlier_mask].reset_index(drop=True)
    removed  = original_len - len(df_clean)
    print(f"[Remove Outliers] {removed} rows removed | "
          f"{len(df_clean)} rows remaining ({removed/original_len*100:.1f}% removed)")
    return df_clean


# =============================================================================
# 🔧 10. UTILITY — OUTLIER SUMMARY REPORT
# =============================================================================

def outlier_summary(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Generates a summary report of outlier counts using Z-Score and IQR methods.

    Args:
        df      : Input DataFrame
        columns : List of numeric columns to inspect

    Returns:
        DataFrame with outlier counts and percentages per column
    """
    rows = []
    for col in columns:
        # Z-Score
        z     = np.abs(stats.zscore(df[col].dropna()))
        z_cnt = (z > 3).sum()

        # IQR
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR    = Q3 - Q1
        iqr_cnt = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()

        rows.append({
            "Column"           : col,
            "Total Rows"       : len(df),
            "Z-Score Outliers" : z_cnt,
            "Z-Score %"        : round(z_cnt / len(df) * 100, 2),
            "IQR Outliers"     : iqr_cnt,
            "IQR %"            : round(iqr_cnt / len(df) * 100, 2),
            "Min"              : round(df[col].min(), 2),
            "Max"              : round(df[col].max(), 2),
            "Skewness"         : round(df[col].skew(), 4),
        })

    report = pd.DataFrame(rows)
    print(report.to_string(index=False))
    return report


# =============================================================================
# 🚀 MAIN — Demo with Synthetic Data
# =============================================================================

if __name__ == "__main__":

    # ── Sample Dataset ──────────────────────────────────────────────────────
    np.random.seed(42)
    n = 200

    data = {
        "Age"    : np.concatenate([np.random.randint(20, 60, n - 5),
                                   [150, -10, 200, 180, 999]]),        # injected outliers
        "Salary" : np.concatenate([np.random.randint(30_000, 150_000, n - 5),
                                   [5_000_000, -1000, 4_500_000, 3_000_000, 0]]),
        "Score"  : np.random.uniform(0, 100, n),
        "Target" : np.random.randint(0, 2, n),
    }
    df = pd.DataFrame(data)
    feature_cols = ["Age", "Salary", "Score"]

    print("=" * 65)
    print("📊 Original Dataset — First 5 Rows")
    print("=" * 65)
    print(df.head())

    # ── Outlier Summary Report ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("📋 Outlier Summary Report")
    print("=" * 65)
    outlier_summary(df, feature_cols)

    # ── 1. Z-Score ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("1️⃣  Z-Score Method")
    print("=" * 65)
    df_z = detect_zscore(df.copy(), feature_cols, threshold=3.0)

    # ── 2. IQR ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("2️⃣  IQR Method")
    print("=" * 65)
    df_iqr = detect_iqr(df.copy(), feature_cols, factor=1.5)

    # ── 3. MAD ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("3️⃣  Modified Z-Score (MAD)")
    print("=" * 65)
    df_mad = detect_modified_zscore(df.copy(), feature_cols, threshold=3.5)

    # ── 4. Isolation Forest ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("4️⃣  Isolation Forest")
    print("=" * 65)
    X = df[feature_cols]
    if_labels, if_scores = detect_isolation_forest(X, contamination=0.05)

    # ── 5. LOF ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("5️⃣  Local Outlier Factor (LOF)")
    print("=" * 65)
    lof_labels, lof_scores = detect_lof(X, n_neighbors=20, contamination=0.05)

    # ── 6. DBSCAN ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("6️⃣  DBSCAN")
    print("=" * 65)
    db_labels = detect_dbscan(X, eps=0.5, min_samples=5)

    # ── 7. Winsorization ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("7️⃣  Winsorization (Capping)")
    print("=" * 65)
    df_wins = apply_winsorization(df.copy(), feature_cols, lower_pct=0.05, upper_pct=0.95)

    # ── 8. Remove IQR Outliers ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("🗑️   Remove IQR Outliers")
    print("=" * 65)
    combined_mask = (
        df_iqr["Age_iqr_outlier"] |
        df_iqr["Salary_iqr_outlier"]
    )
    df_clean = remove_outliers(df, combined_mask)

    print("\n✅ All outlier detection techniques demonstrated successfully!")
