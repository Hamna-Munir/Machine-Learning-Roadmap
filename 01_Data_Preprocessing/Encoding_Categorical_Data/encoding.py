# =============================================================================
# 📦 Encoding Categorical Data — Reusable ML Script
# =============================================================================
# Author   : Hamna Munir
# Topic    : 01_Data_Preprocessing / Encoding_Categorical_Data
# File     : encoding.py
# =============================================================================

# -----------------------------------------------------------------------------
# 📚 Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import KFold
from category_encoders import BinaryEncoder, TargetEncoder, CountEncoder, HashingEncoder

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 🔧 1. LABEL ENCODING
# =============================================================================

def apply_label_encoding(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Applies Label Encoding to specified columns.

    Best for:
        - Ordinal features
        - Tree-based models

    Args:
        df      : Input DataFrame
        columns : List of column names to encode

    Returns:
        DataFrame with label-encoded columns
    """
    df = df.copy()
    le = LabelEncoder()

    for col in columns:
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"[LabelEncoder] '{col}' → classes: {list(le.classes_)}")

    return df


# =============================================================================
# 🔧 2. ONE-HOT ENCODING
# =============================================================================

def apply_ohe(df: pd.DataFrame, columns: list, drop_first: bool = True) -> pd.DataFrame:
    """
    Applies One-Hot Encoding using pandas get_dummies().

    Best for:
        - Nominal features with low cardinality
        - Linear models, Logistic Regression, SVMs

    Args:
        df         : Input DataFrame
        columns    : List of column names to encode
        drop_first : Drop first dummy to avoid multicollinearity (default: True)

    Returns:
        DataFrame with one-hot encoded columns
    """
    df = df.copy()
    df = pd.get_dummies(df, columns=columns, drop_first=drop_first, dtype=int)
    print(f"[OneHotEncoding] New shape: {df.shape}")
    return df


def apply_ohe_sklearn(X_train: pd.DataFrame,
                       X_test: pd.DataFrame,
                       columns: list) -> tuple:
    """
    Applies sklearn OneHotEncoder (fit on train, transform on test).

    Args:
        X_train : Training features
        X_test  : Test features
        columns : Columns to encode

    Returns:
        Tuple of (X_train_encoded, X_test_encoded) as DataFrames
    """
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")

    train_encoded = encoder.fit_transform(X_train[columns])
    test_encoded  = encoder.transform(X_test[columns])

    feature_names = encoder.get_feature_names_out(columns)

    train_df = pd.DataFrame(train_encoded, columns=feature_names, index=X_train.index)
    test_df  = pd.DataFrame(test_encoded,  columns=feature_names, index=X_test.index)

    X_train = X_train.drop(columns=columns).join(train_df)
    X_test  = X_test.drop(columns=columns).join(test_df)

    print(f"[sklearn OHE] Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    return X_train, X_test


# =============================================================================
# 🔧 3. ORDINAL ENCODING
# =============================================================================

def apply_ordinal_encoding(df: pd.DataFrame,
                            column: str,
                            order: list) -> pd.DataFrame:
    """
    Applies Ordinal Encoding with a user-defined category order.

    Best for:
        - Explicitly ordered categorical features

    Args:
        df     : Input DataFrame
        column : Column name to encode
        order  : Ordered list of categories (low → high)

    Returns:
        DataFrame with ordinal-encoded column

    Example:
        apply_ordinal_encoding(df, 'Education', ['High School', "Bachelor's", "Master's", 'PhD'])
    """
    df = df.copy()
    encoder = OrdinalEncoder(categories=[order])
    df[column] = encoder.fit_transform(df[[column]])
    print(f"[OrdinalEncoder] '{column}' order: {order}")
    return df


# =============================================================================
# 🔧 4. BINARY ENCODING
# =============================================================================

def apply_binary_encoding(X_train: pd.DataFrame,
                           X_test: pd.DataFrame,
                           columns: list) -> tuple:
    """
    Applies Binary Encoding using category_encoders.

    Best for:
        - High cardinality nominal features

    Args:
        X_train : Training features
        X_test  : Test features
        columns : Columns to encode

    Returns:
        Tuple of (X_train_encoded, X_test_encoded)
    """
    encoder = BinaryEncoder(cols=columns)
    X_train = encoder.fit_transform(X_train)
    X_test  = encoder.transform(X_test)
    print(f"[BinaryEncoder] Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    return X_train, X_test


# =============================================================================
# 🔧 5. FREQUENCY / COUNT ENCODING
# =============================================================================

def apply_frequency_encoding(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Replaces categories with their frequency count in the dataset.

    Best for:
        - High cardinality features where frequency is meaningful

    Args:
        df      : Input DataFrame
        columns : List of column names to encode

    Returns:
        DataFrame with frequency-encoded columns
    """
    df = df.copy()
    for col in columns:
        freq_map = df[col].value_counts().to_dict()
        df[col] = df[col].map(freq_map)
        print(f"[FrequencyEncoder] '{col}' → top values: {dict(list(freq_map.items())[:5])}")
    return df


# =============================================================================
# 🔧 6. TARGET ENCODING (with K-Fold to prevent leakage)
# =============================================================================

def apply_target_encoding_kfold(df: pd.DataFrame,
                                  column: str,
                                  target: str,
                                  n_splits: int = 5) -> pd.DataFrame:
    """
    Applies Target Encoding with K-Fold cross-validation to prevent data leakage.

    Best for:
        - High cardinality features with strong category-target relationship

    Args:
        df       : Input DataFrame (train only)
        column   : Column name to encode
        target   : Target column name
        n_splits : Number of CV folds (default: 5)

    Returns:
        DataFrame with target-encoded column (no leakage)
    """
    df = df.copy()
    df[f"{column}_target_enc"] = np.nan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(df):
        train_fold = df.iloc[train_idx]
        mean_map = train_fold.groupby(column)[target].mean()
        df.iloc[val_idx, df.columns.get_loc(f"{column}_target_enc")] = (
            df.iloc[val_idx][column].map(mean_map)
        )

    global_mean = df[target].mean()
    df[f"{column}_target_enc"] = df[f"{column}_target_enc"].fillna(global_mean)

    print(f"[TargetEncoder (KFold)] '{column}' encoded with {n_splits}-fold CV")
    return df


# =============================================================================
# 🔧 7. HASHING ENCODING
# =============================================================================

def apply_hashing_encoding(X_train: pd.DataFrame,
                             X_test: pd.DataFrame,
                             columns: list,
                             n_components: int = 8) -> tuple:
    """
    Applies Feature Hashing (Hashing Trick) Encoding.

    Best for:
        - Very high cardinality features
        - Memory-efficient encoding

    Args:
        X_train      : Training features
        X_test       : Test features
        columns      : Columns to encode
        n_components : Number of output hash columns (default: 8)

    Returns:
        Tuple of (X_train_encoded, X_test_encoded)
    """
    encoder = HashingEncoder(cols=columns, n_components=n_components)
    X_train = encoder.fit_transform(X_train)
    X_test  = encoder.transform(X_test)
    print(f"[HashingEncoder] n_components={n_components} | Train shape: {X_train.shape}")
    return X_train, X_test


# =============================================================================
# 🔧 8. UTILITY — IDENTIFY CATEGORICAL COLUMNS
# =============================================================================

def get_categorical_columns(df: pd.DataFrame,
                              max_cardinality: int = None) -> dict:
    """
    Identifies and categorizes categorical columns in a DataFrame.

    Args:
        df              : Input DataFrame
        max_cardinality : Optional threshold to separate low vs high cardinality

    Returns:
        Dictionary with keys: 'all', 'low_cardinality', 'high_cardinality'
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    result = {"all": cat_cols}

    if max_cardinality:
        low  = [c for c in cat_cols if df[c].nunique() <= max_cardinality]
        high = [c for c in cat_cols if df[c].nunique() >  max_cardinality]
        result["low_cardinality"]  = low
        result["high_cardinality"] = high
        print(f"[Categorical Columns]")
        print(f"  Low cardinality  (≤ {max_cardinality}): {low}")
        print(f"  High cardinality (> {max_cardinality}): {high}")
    else:
        print(f"[Categorical Columns] {cat_cols}")

    return result


# =============================================================================
# 🔧 9. UTILITY — CARDINALITY REPORT
# =============================================================================

def cardinality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a summary of unique value counts for all categorical columns.

    Args:
        df : Input DataFrame

    Returns:
        DataFrame with cardinality report
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    report = pd.DataFrame({
        "Column"      : cat_cols,
        "Unique Values": [df[c].nunique() for c in cat_cols],
        "Sample Values": [df[c].unique()[:5].tolist() for c in cat_cols],
        "Missing %"   : [round(df[c].isna().mean() * 100, 2) for c in cat_cols]
    }).sort_values("Unique Values", ascending=False).reset_index(drop=True)

    print(report.to_string(index=False))
    return report


# =============================================================================
# 🚀 MAIN — Demo with Synthetic Data
# =============================================================================

if __name__ == "__main__":

    # ── Sample Dataset ──────────────────────────────────────────────────────
    data = {
        "Color"    : ["Red", "Blue", "Green", "Blue", "Red", "Green"],
        "Education": ["Bachelor's", "PhD", "High School", "Master's", "PhD", "Bachelor's"],
        "City"     : ["London", "Paris", "Berlin", "London", "Paris", "Berlin"],
        "Gender"   : ["Male", "Female", "Male", "Female", "Male", "Female"],
        "Price"    : [250, 400, 150, 300, 420, 180],
        "Target"   : [1, 0, 1, 0, 0, 1],
    }
    df = pd.DataFrame(data)

    print("=" * 60)
    print("📊 Original Dataset")
    print("=" * 60)
    print(df)

    # ── Cardinality Report ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📋 Cardinality Report")
    print("=" * 60)
    cardinality_report(df)

    # ── 1. Label Encoding ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("1️⃣  Label Encoding")
    print("=" * 60)
    df_label = apply_label_encoding(df.copy(), columns=["Color", "Gender"])
    print(df_label[["Color", "Gender"]])

    # ── 2. One-Hot Encoding ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2️⃣  One-Hot Encoding")
    print("=" * 60)
    df_ohe = apply_ohe(df.copy(), columns=["Color"], drop_first=True)
    print(df_ohe)

    # ── 3. Ordinal Encoding ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3️⃣  Ordinal Encoding")
    print("=" * 60)
    edu_order = ["High School", "Bachelor's", "Master's", "PhD"]
    df_ord = apply_ordinal_encoding(df.copy(), column="Education", order=edu_order)
    print(df_ord[["Education"]])

    # ── 4. Frequency Encoding ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("5️⃣  Frequency Encoding")
    print("=" * 60)
    df_freq = apply_frequency_encoding(df.copy(), columns=["City"])
    print(df_freq[["City"]])

    # ── 5. Target Encoding (K-Fold) ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("6️⃣  Target Encoding (K-Fold, no leakage)")
    print("=" * 60)
    df_target = apply_target_encoding_kfold(df.copy(), column="City", target="Target")
    print(df_target[["City", "City_target_enc", "Target"]])

    print("\n✅ All encoding techniques demonstrated successfully!")
