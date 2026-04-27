# =============================================================================
# 📦 Feature Engineering — Reusable ML Script
# =============================================================================
# Author   : Hamna Munir
# Topic    : 01_Data_Preprocessing / Feature_Engineering
# File     : feature_engineering.py
# =============================================================================

# -----------------------------------------------------------------------------
# 📚 Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 🔧 1. MATHEMATICAL FEATURE CREATION
# =============================================================================

def create_ratio_features(df: pd.DataFrame,
                            pairs: list,
                            epsilon: float = 1e-10) -> pd.DataFrame:
    """
    Creates ratio features from pairs of numeric columns.

    Formula : new_feature = numerator / (denominator + ε)

    Best for:
        - Capturing relative relationships (CTR, profit margin, BMI)
        - Normalizing one metric by another

    Args:
        df      : Input DataFrame
        pairs   : List of tuples [(numerator_col, denominator_col, new_name), ...]
        epsilon : Small constant to avoid division by zero (default: 1e-10)

    Returns:
        DataFrame with ratio features added

    Example:
        create_ratio_features(df, [('Revenue', 'Expenses', 'Profit_Ratio')])
    """
    df = df.copy()
    for numerator, denominator, new_name in pairs:
        df[new_name] = df[numerator] / (df[denominator] + epsilon)
        print(f"[Ratio Feature] '{new_name}' = {numerator} / {denominator}")
    return df


def create_difference_features(df: pd.DataFrame,
                                 pairs: list) -> pd.DataFrame:
    """
    Creates difference features from pairs of numeric columns.

    Formula : new_feature = col_a - col_b

    Best for:
        - Capturing improvement or gap (score_post - score_pre)
        - Time gaps (delivery_date - order_date)

    Args:
        df    : Input DataFrame
        pairs : List of tuples [(col_a, col_b, new_name), ...]

    Returns:
        DataFrame with difference features added
    """
    df = df.copy()
    for col_a, col_b, new_name in pairs:
        df[new_name] = df[col_a] - df[col_b]
        print(f"[Difference Feature] '{new_name}' = {col_a} - {col_b}")
    return df


def create_product_features(df: pd.DataFrame,
                              pairs: list) -> pd.DataFrame:
    """
    Creates product (multiplication) features from pairs of numeric columns.

    Formula : new_feature = col_a × col_b

    Best for:
        - Manual interaction terms
        - Capturing combined effect of two features

    Args:
        df    : Input DataFrame
        pairs : List of tuples [(col_a, col_b, new_name), ...]

    Returns:
        DataFrame with product features added
    """
    df = df.copy()
    for col_a, col_b, new_name in pairs:
        df[new_name] = df[col_a] * df[col_b]
        print(f"[Product Feature] '{new_name}' = {col_a} × {col_b}")
    return df


# =============================================================================
# 🔧 2. TRANSFORMATION FEATURES
# =============================================================================

def apply_log_features(df: pd.DataFrame,
                         columns: list,
                         suffix: str = "_log") -> pd.DataFrame:
    """
    Applies log(X + 1) transformation to reduce right skewness.

    Formula : X_new = log(X + 1)

    Best for:
        - Right-skewed distributions (income, prices, counts)
        - Positive-valued features only

    Args:
        df      : Input DataFrame
        columns : List of numeric columns to transform
        suffix  : Suffix for new column names (default: '_log')

    Returns:
        DataFrame with log-transformed columns added (original preserved)
    """
    df = df.copy()
    for col in columns:
        if (df[col] < 0).any():
            print(f"[LogFeature] ⚠️  '{col}' has negative values — skipping")
            continue
        skew_before = round(df[col].skew(), 4)
        df[col + suffix] = np.log1p(df[col])
        skew_after = round(df[col + suffix].skew(), 4)
        print(f"[LogFeature] '{col}{suffix}' | skew: {skew_before} → {skew_after}")
    return df


def apply_sqrt_features(df: pd.DataFrame,
                          columns: list,
                          suffix: str = "_sqrt") -> pd.DataFrame:
    """
    Applies square root transformation to moderate right skewness.

    Formula : X_new = √X

    Args:
        df      : Input DataFrame
        columns : List of non-negative numeric columns
        suffix  : Suffix for new column names (default: '_sqrt')

    Returns:
        DataFrame with sqrt-transformed columns added
    """
    df = df.copy()
    for col in columns:
        if (df[col] < 0).any():
            print(f"[SqrtFeature] ⚠️  '{col}' has negative values — skipping")
            continue
        df[col + suffix] = np.sqrt(df[col])
        print(f"[SqrtFeature] '{col}{suffix}' created")
    return df


def apply_power_features(df: pd.DataFrame,
                           columns: list,
                           power: float = 2,
                           suffix: str = None) -> pd.DataFrame:
    """
    Raises features to a specified power to capture non-linear patterns.

    Formula : X_new = X ^ power

    Args:
        df      : Input DataFrame
        columns : List of numeric columns
        power   : Exponent value (default: 2 for squared)
        suffix  : Suffix for new column names (auto-generated if None)

    Returns:
        DataFrame with power-transformed columns added
    """
    df = df.copy()
    sfx = suffix if suffix else f"_pow{int(power)}"
    for col in columns:
        df[col + sfx] = df[col] ** power
        print(f"[PowerFeature] '{col}{sfx}' = {col}^{power}")
    return df


# =============================================================================
# 🔧 3. POLYNOMIAL & INTERACTION FEATURES
# =============================================================================

def create_polynomial_features(X_train: pd.DataFrame,
                                  X_test: pd.DataFrame,
                                  columns: list,
                                  degree: int = 2,
                                  interaction_only: bool = False,
                                  include_bias: bool = False) -> tuple:
    """
    Generates polynomial and interaction features using sklearn.

    Formula (degree=2, [a, b]):
        → [a, b, a², a·b, b²]

    Best for:
        - Capturing non-linear relationships
        - Linear models on non-linear data

    ⚠️ Warning: Can cause dimensionality explosion with many features.
        Use interaction_only=True to limit to cross-product terms only.

    Args:
        X_train          : Training DataFrame
        X_test           : Test DataFrame
        columns          : Columns to expand
        degree           : Polynomial degree (default: 2)
        interaction_only : Only create cross terms, no powers (default: False)
        include_bias     : Include a bias (ones) column (default: False)

    Returns:
        Tuple of (X_train_expanded, X_test_expanded) as DataFrames
    """
    poly = PolynomialFeatures(
        degree=degree,
        interaction_only=interaction_only,
        include_bias=include_bias
    )

    train_poly = poly.fit_transform(X_train[columns])
    test_poly  = poly.transform(X_test[columns])

    feature_names  = poly.get_feature_names_out(columns)
    train_poly_df  = pd.DataFrame(train_poly, columns=feature_names, index=X_train.index)
    test_poly_df   = pd.DataFrame(test_poly,  columns=feature_names, index=X_test.index)

    X_train_out = X_train.drop(columns=columns).join(train_poly_df)
    X_test_out  = X_test.drop(columns=columns).join(test_poly_df)

    new_features = [f for f in feature_names if f not in columns]
    print(f"[PolynomialFeatures] degree={degree} | "
          f"{len(columns)} → {len(feature_names)} features | "
          f"New: {new_features}")

    return X_train_out, X_test_out


# =============================================================================
# 🔧 4. DATE & TIME FEATURES
# =============================================================================

def extract_datetime_features(df: pd.DataFrame,
                                date_col: str,
                                drop_original: bool = False,
                                cyclic_encode: bool = True) -> pd.DataFrame:
    """
    Extracts rich temporal features from a datetime column.

    Features extracted:
        Year, Month, Day, DayOfWeek, DayOfYear,
        Quarter, WeekOfYear, IsWeekend, IsMonthStart, IsMonthEnd,
        Hour (if timestamp), Days_Since_Min

    Cyclic encoding (optional):
        Month_sin / Month_cos, DayOfWeek_sin / DayOfWeek_cos,
        Hour_sin / Hour_cos (captures circular nature of time)

    Args:
        df             : Input DataFrame
        date_col       : Name of the datetime column
        drop_original  : Drop the original datetime column (default: False)
        cyclic_encode  : Add sine/cosine cyclic features (default: True)

    Returns:
        DataFrame with temporal features added
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    dt = df[date_col].dt

    # ── Basic Components ─────────────────────────────────────────────────
    df[f"{date_col}_Year"]        = dt.year
    df[f"{date_col}_Month"]       = dt.month
    df[f"{date_col}_Day"]         = dt.day
    df[f"{date_col}_DayOfWeek"]   = dt.dayofweek          # 0=Monday, 6=Sunday
    df[f"{date_col}_DayOfYear"]   = dt.dayofyear
    df[f"{date_col}_Quarter"]     = dt.quarter
    df[f"{date_col}_WeekOfYear"]  = dt.isocalendar().week.astype(int)
    df[f"{date_col}_IsWeekend"]   = (dt.dayofweek >= 5).astype(int)
    df[f"{date_col}_IsMonthStart"]= dt.is_month_start.astype(int)
    df[f"{date_col}_IsMonthEnd"]  = dt.is_month_end.astype(int)
    df[f"{date_col}_Days_Since"]  = (dt - dt.min()).dt.days

    # ── Hour (only if time component exists) ─────────────────────────────
    if dt.hour.nunique() > 1:
        df[f"{date_col}_Hour"]        = dt.hour
        df[f"{date_col}_IsBusinessHr"]= ((dt.hour >= 9) & (dt.hour < 18)).astype(int)

    # ── Cyclic Encoding ───────────────────────────────────────────────────
    if cyclic_encode:
        df[f"{date_col}_Month_sin"]     = np.sin(2 * np.pi * dt.month     / 12)
        df[f"{date_col}_Month_cos"]     = np.cos(2 * np.pi * dt.month     / 12)
        df[f"{date_col}_DayOfWeek_sin"] = np.sin(2 * np.pi * dt.dayofweek / 7)
        df[f"{date_col}_DayOfWeek_cos"] = np.cos(2 * np.pi * dt.dayofweek / 7)

    if drop_original:
        df.drop(columns=[date_col], inplace=True)

    new_cols = [c for c in df.columns if c.startswith(date_col + "_")]
    print(f"[DatetimeFeatures] '{date_col}' → {len(new_cols)} new features: {new_cols}")
    return df


# =============================================================================
# 🔧 5. TEXT-BASED FEATURES
# =============================================================================

def extract_text_features(df: pd.DataFrame,
                            text_col: str) -> pd.DataFrame:
    """
    Extracts statistical and structural features from a text/string column.

    Features created:
        - char_length    : Number of characters
        - word_count     : Number of words
        - unique_words   : Number of unique words
        - avg_word_len   : Average word length
        - num_digits     : Count of digit characters
        - num_uppercase  : Count of uppercase letters
        - has_special    : Has special characters (binary)
        - has_url        : Contains URL pattern (binary)
        - has_email      : Contains email pattern (binary)
        - is_empty       : Is null or empty string (binary)

    Args:
        df       : Input DataFrame
        text_col : Name of the string/text column

    Returns:
        DataFrame with text-derived features added
    """
    df  = df.copy()
    col = df[text_col].astype(str).fillna("")

    df[f"{text_col}_char_length"]   = col.str.len()
    df[f"{text_col}_word_count"]    = col.str.split().str.len()
    df[f"{text_col}_unique_words"]  = col.apply(lambda x: len(set(x.lower().split())))
    df[f"{text_col}_avg_word_len"]  = col.apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
    )
    df[f"{text_col}_num_digits"]    = col.str.count(r"\d")
    df[f"{text_col}_num_uppercase"] = col.str.count(r"[A-Z]")
    df[f"{text_col}_has_special"]   = col.str.contains(r"[^a-zA-Z0-9\s]").astype(int)
    df[f"{text_col}_has_url"]       = col.str.contains(
        r"http[s]?://|www\.", regex=True).astype(int)
    df[f"{text_col}_has_email"]     = col.str.contains(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", regex=True).astype(int)
    df[f"{text_col}_is_empty"]      = (col.str.strip() == "").astype(int)

    new_cols = [c for c in df.columns if c.startswith(text_col + "_")]
    print(f"[TextFeatures] '{text_col}' → {len(new_cols)} new features")
    return df


# =============================================================================
# 🔧 6. AGGREGATION / GROUP-LEVEL FEATURES
# =============================================================================

def create_group_features(train_df: pd.DataFrame,
                            test_df: pd.DataFrame,
                            group_col: str,
                            agg_col: str,
                            agg_funcs: list = None) -> tuple:
    """
    Creates group-level aggregation features.

    ⚠️ IMPORTANT: Fit on TRAIN only → apply to both train and test.
    Applying on full data causes data leakage.

    Features created (example — group='City', agg='Salary'):
        City_Salary_mean, City_Salary_median, City_Salary_std,
        City_Salary_max,  City_Salary_min,    City_Salary_count,
        Salary_vs_City_mean  (deviation from group mean)

    Args:
        train_df   : Training DataFrame
        test_df    : Test DataFrame
        group_col  : Column to group by (e.g., 'City', 'Category')
        agg_col    : Column to aggregate (e.g., 'Salary', 'Price')
        agg_funcs  : List of aggregation functions (default: mean, median, std, max, min, count)

    Returns:
        Tuple of (train_df_with_features, test_df_with_features)
    """
    if agg_funcs is None:
        agg_funcs = ["mean", "median", "std", "max", "min", "count"]

    train_df = train_df.copy()
    test_df  = test_df.copy()

    prefix  = f"{group_col}_{agg_col}"
    grouped = train_df.groupby(group_col)[agg_col]

    for func in agg_funcs:
        col_name = f"{prefix}_{func}"
        stat_map = grouped.agg(func)
        train_df[col_name] = train_df[group_col].map(stat_map)
        test_df[col_name]  = test_df[group_col].map(stat_map)

    # Deviation from group mean
    dev_col = f"{agg_col}_vs_{group_col}_mean"
    train_df[dev_col] = train_df[agg_col] - train_df[f"{prefix}_mean"]
    test_df[dev_col]  = test_df[agg_col]  - test_df[f"{prefix}_mean"]

    new_cols = [c for c in train_df.columns
                if c.startswith(prefix) or c == dev_col]
    print(f"[GroupFeatures] group='{group_col}', agg='{agg_col}' → "
          f"{len(new_cols)} new features: {new_cols}")

    return train_df, test_df


# =============================================================================
# 🔧 7. BINNING / DISCRETIZATION
# =============================================================================

def apply_equal_width_binning(df: pd.DataFrame,
                                column: str,
                                n_bins: int,
                                labels: list = None,
                                new_col: str = None) -> pd.DataFrame:
    """
    Applies equal-width binning (pd.cut) to a numeric column.

    Best for:
        - Uniform-ish distributions
        - When bin width matters more than bin size

    Args:
        df      : Input DataFrame
        column  : Numeric column to bin
        n_bins  : Number of equal-width bins
        labels  : Optional list of bin labels
        new_col : Name for the new binned column (default: '{column}_bin')

    Returns:
        DataFrame with binned column added
    """
    df      = df.copy()
    new_col = new_col or f"{column}_bin"
    df[new_col] = pd.cut(df[column], bins=n_bins, labels=labels)
    print(f"[EqualWidthBin] '{column}' → '{new_col}' with {n_bins} bins")
    return df


def apply_quantile_binning(df: pd.DataFrame,
                             column: str,
                             q: int,
                             labels: list = None,
                             new_col: str = None) -> pd.DataFrame:
    """
    Applies equal-frequency binning (pd.qcut) to a numeric column.

    Best for:
        - Skewed distributions
        - When equal-sized bins matter

    Args:
        df      : Input DataFrame
        column  : Numeric column to bin
        q       : Number of quantile bins (e.g., 4 = quartiles)
        labels  : Optional list of bin labels
        new_col : Name for the new binned column (default: '{column}_qbin')

    Returns:
        DataFrame with quantile-binned column added
    """
    df      = df.copy()
    new_col = new_col or f"{column}_qbin"
    df[new_col] = pd.qcut(df[column], q=q, labels=labels, duplicates="drop")
    print(f"[QuantileBin] '{column}' → '{new_col}' with {q} quantile bins")
    return df


def apply_custom_binning(df: pd.DataFrame,
                          column: str,
                          bins: list,
                          labels: list,
                          new_col: str = None) -> pd.DataFrame:
    """
    Applies custom threshold-based binning using domain knowledge.

    Args:
        df      : Input DataFrame
        column  : Numeric column to bin
        bins    : List of bin edge values (e.g., [0, 18, 35, 60, 100])
        labels  : List of bin labels (len = len(bins) - 1)
        new_col : Name for the new binned column

    Returns:
        DataFrame with custom-binned column added

    Example:
        apply_custom_binning(df, 'Age', [0,18,35,60,100],
                             ['Minor','Young','Adult','Senior'])
    """
    df      = df.copy()
    new_col = new_col or f"{column}_custom_bin"
    df[new_col] = pd.cut(df[column], bins=bins, labels=labels, include_lowest=True)
    print(f"[CustomBin] '{column}' → '{new_col}' | bins: {bins}")
    return df


# =============================================================================
# 🔧 8. FLAG / INDICATOR FEATURES
# =============================================================================

def create_flag_features(df: pd.DataFrame,
                          flags: list) -> pd.DataFrame:
    """
    Creates binary flag (indicator) features from conditions.

    Args:
        df    : Input DataFrame
        flags : List of tuples [(condition_series, new_col_name), ...]

    Returns:
        DataFrame with binary flag columns added (0 or 1)

    Example:
        create_flag_features(df, [
            (df['Salary'] > 100_000, 'Is_High_Earner'),
            (df['Age'] > 60,         'Is_Senior'),
            (df['Email'].isna(),     'Is_Missing_Email'),
        ])
    """
    df = df.copy()
    for condition, col_name in flags:
        df[col_name] = condition.astype(int)
        count = df[col_name].sum()
        print(f"[FlagFeature] '{col_name}' → {count} rows flagged "
              f"({count/len(df)*100:.1f}%)")
    return df


def create_missing_indicators(df: pd.DataFrame,
                                columns: list = None) -> pd.DataFrame:
    """
    Creates binary indicator features for missing values.

    Best for:
        - When missingness itself carries signal (not random)
        - Before imputation — preserves the pattern

    Args:
        df      : Input DataFrame
        columns : Columns to create indicators for (default: all with missing values)

    Returns:
        DataFrame with '_is_missing' indicator columns added
    """
    df = df.copy()
    if columns is None:
        columns = [c for c in df.columns if df[c].isna().any()]

    for col in columns:
        new_col = f"{col}_is_missing"
        df[new_col] = df[col].isna().astype(int)
        count = df[new_col].sum()
        print(f"[MissingIndicator] '{new_col}' → {count} missing rows")

    return df


# =============================================================================
# 🔧 9. UTILITY — FEATURE CORRELATION WITH TARGET
# =============================================================================

def feature_target_correlation(df: pd.DataFrame,
                                 target: str,
                                 top_n: int = 20) -> pd.DataFrame:
    """
    Computes and ranks correlations of all numeric features with the target.

    Args:
        df     : DataFrame with features and target
        target : Target column name
        top_n  : Number of top features to display (default: 20)

    Returns:
        DataFrame sorted by absolute correlation with target
    """
    num_df = df.select_dtypes(include=[np.number])
    corr   = num_df.corr()[target].drop(target).abs().sort_values(ascending=False)

    report = pd.DataFrame({
        "Feature"    : corr.index,
        "|Correlation|": corr.values.round(4),
        "Direction"  : ["Positive" if v > 0 else "Negative"
                        for v in num_df.corr()[target].drop(target)[corr.index]]
    }).head(top_n)

    print(f"\n📊 Top {top_n} Features by |Correlation| with '{target}':")
    print(report.to_string(index=False))
    return report


# =============================================================================
# 🔧 10. UTILITY — FEATURE SUMMARY REPORT
# =============================================================================

def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a summary report of all features in the DataFrame.

    Args:
        df : Input DataFrame

    Returns:
        DataFrame with dtype, missing %, unique count, and sample values
    """
    report = pd.DataFrame({
        "Column"       : df.columns,
        "Dtype"        : df.dtypes.values,
        "Non-Null"     : df.count().values,
        "Missing %"    : (df.isna().mean() * 100).round(2).values,
        "Unique Values": df.nunique().values,
        "Sample"       : [df[c].dropna().iloc[0] if df[c].notna().any() else None
                          for c in df.columns],
    })
    print(report.to_string(index=False))
    return report


# =============================================================================
# 🚀 MAIN — Demo with Synthetic Dataset
# =============================================================================

if __name__ == "__main__":

    # ── Sample Dataset ──────────────────────────────────────────────────────
    np.random.seed(42)
    n = 300

    data = {
        "Age"         : np.random.randint(20, 65, n),
        "Salary"      : np.random.randint(30_000, 150_000, n),
        "Expenses"    : np.random.randint(10_000, 80_000, n),
        "Score_Pre"   : np.random.randint(40, 80, n),
        "Score_Post"  : np.random.randint(50, 100, n),
        "City"        : np.random.choice(["London", "Paris", "Berlin", "Tokyo"], n),
        "JoinDate"    : pd.date_range("2018-01-01", periods=n, freq="3D"),
        "Description" : ["Good product review!" if i % 2 == 0
                         else "Bad experience, will not buy again." for i in range(n)],
        "Income"      : np.random.exponential(50_000, n),
        "Target"      : np.random.randint(0, 2, n),
    }
    df = pd.DataFrame(data)

    print("=" * 65)
    print("📊 Original Dataset — First 5 Rows")
    print("=" * 65)
    print(df.head())

    # ── Train-Test Split (always before engineering!) ─────────────────────
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    # ── 1. Ratio Features ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("1️⃣  Ratio Features")
    print("=" * 65)
    train_df = create_ratio_features(
        train_df,
        pairs=[("Salary", "Expenses", "Savings_Rate"),
               ("Salary", "Age",      "Salary_Per_Age")]
    )
    test_df = create_ratio_features(
        test_df,
        pairs=[("Salary", "Expenses", "Savings_Rate"),
               ("Salary", "Age",      "Salary_Per_Age")]
    )

    # ── 2. Difference Features ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("2️⃣  Difference Features")
    print("=" * 65)
    train_df = create_difference_features(
        train_df, [("Score_Post", "Score_Pre", "Score_Improvement")]
    )
    test_df = create_difference_features(
        test_df, [("Score_Post", "Score_Pre", "Score_Improvement")]
    )

    # ── 3. Log Transform ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("3️⃣  Log Transformation")
    print("=" * 65)
    train_df = apply_log_features(train_df, columns=["Income", "Salary"])
    test_df  = apply_log_features(test_df,  columns=["Income", "Salary"])

    # ── 4. Polynomial Features ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("4️⃣  Polynomial Features (degree=2)")
    print("=" * 65)
    X_train_num = train_df[["Age", "Score_Pre"]].copy()
    X_test_num  = test_df[["Age", "Score_Pre"]].copy()
    X_train_poly, X_test_poly = create_polynomial_features(
        X_train_num, X_test_num, columns=["Age", "Score_Pre"], degree=2
    )

    # ── 5. Datetime Features ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("5️⃣  Datetime Features")
    print("=" * 65)
    train_df = extract_datetime_features(train_df, "JoinDate", cyclic_encode=True)
    test_df  = extract_datetime_features(test_df,  "JoinDate", cyclic_encode=True)

    # ── 6. Text Features ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("6️⃣  Text Features")
    print("=" * 65)
    train_df = extract_text_features(train_df, "Description")
    test_df  = extract_text_features(test_df,  "Description")

    # ── 7. Group Aggregation Features ─────────────────────────────────────
    print("\n" + "=" * 65)
    print("7️⃣  Group Aggregation Features")
    print("=" * 65)
    train_df, test_df = create_group_features(
        train_df, test_df, group_col="City", agg_col="Salary"
    )

    # ── 8. Binning ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("8️⃣  Binning / Discretization")
    print("=" * 65)
    train_df = apply_custom_binning(
        train_df, "Age",
        bins=[0, 30, 45, 60, 100],
        labels=["Young", "Mid", "Senior", "Elder"]
    )
    test_df = apply_custom_binning(
        test_df, "Age",
        bins=[0, 30, 45, 60, 100],
        labels=["Young", "Mid", "Senior", "Elder"]
    )

    # ── 9. Flag Features ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("9️⃣  Flag / Indicator Features")
    print("=" * 65)
    train_df = create_flag_features(train_df, [
        (train_df["Salary"] > 100_000, "Is_High_Earner"),
        (train_df["Score_Improvement"] > 0, "Score_Improved"),
    ])
    test_df = create_flag_features(test_df, [
        (test_df["Salary"] > 100_000, "Is_High_Earner"),
        (test_df["Score_Improvement"] > 0, "Score_Improved"),
    ])

    # ── 10. Feature Summary ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("📋 Final Feature Summary")
    print("=" * 65)
    feature_summary(train_df)

    print(f"\n✅ Feature Engineering complete!")
    print(f"   Original features : {df.shape[1]}")
    print(f"   Engineered train  : {train_df.shape[1]} features")
