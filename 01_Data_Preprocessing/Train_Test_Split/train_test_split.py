# =============================================================================
# 📦 Train-Test Split — Reusable ML Script
# =============================================================================
# Author   : Hamna Munir
# Topic    : 01_Data_Preprocessing / Train_Test_Split
# File     : train_test_split.py
# =============================================================================

# -----------------------------------------------------------------------------
# 📚 Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    GroupKFold,
    LeaveOneOut,
    cross_val_score,
    cross_validate,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 🔧 1. SIMPLE RANDOM SPLIT
# =============================================================================

def simple_split(X: pd.DataFrame,
                  y: pd.Series,
                  test_size: float = 0.2,
                  random_state: int = 42) -> tuple:
    """
    Splits data into train and test sets using random shuffling.

    Best for:
        - Large datasets
        - IID (independently and identically distributed) data
        - Regression or balanced classification

    Args:
        X            : Feature DataFrame
        y            : Target Series
        test_size    : Proportion for test set (default: 0.2)
        random_state : Reproducibility seed (default: 42)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=True, random_state=random_state
    )
    _split_report("Simple Random Split", X, X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test


# =============================================================================
# 🔧 2. STRATIFIED SPLIT
# =============================================================================

def stratified_split(X: pd.DataFrame,
                      y: pd.Series,
                      test_size: float = 0.2,
                      random_state: int = 42) -> tuple:
    """
    Splits data preserving the class proportion of the target variable.

    Best for:
        - Classification tasks — ALWAYS use over simple split
        - Imbalanced datasets — preserves minority class ratio
        - Multi-class problems

    Args:
        X            : Feature DataFrame
        y            : Target Series (categorical/binary)
        test_size    : Proportion for test set (default: 0.2)
        random_state : Reproducibility seed (default: 42)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    _split_report("Stratified Split", X, X_train, X_test, y_train, y_test)

    # Class distribution check
    print(f"  Train class dist : {dict(y_train.value_counts(normalize=True).round(3))}")
    print(f"  Test  class dist : {dict(y_test.value_counts(normalize=True).round(3))}")

    return X_train, X_test, y_train, y_test


# =============================================================================
# 🔧 3. THREE-WAY SPLIT (Train / Validation / Test)
# =============================================================================

def three_way_split(X: pd.DataFrame,
                     y: pd.Series,
                     val_size: float = 0.15,
                     test_size: float = 0.15,
                     stratify: bool = True,
                     random_state: int = 42) -> tuple:
    """
    Splits data into Train / Validation / Test sets.

    Split logic:
        1. Split off test set first
        2. Split remaining data into train and validation

    Best for:
        - Hyperparameter tuning (use val set) + final evaluation (use test set)
        - Deep learning pipelines
        - Any project where the test set must remain untouched

    Args:
        X            : Feature DataFrame
        y            : Target Series
        val_size     : Proportion for validation set (default: 0.15)
        test_size    : Proportion for test set (default: 0.15)
        stratify     : Preserve class proportions (default: True)
        random_state : Reproducibility seed (default: 42)

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    strat = y if stratify else None

    # Step 1: split off test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size,
        stratify=strat, random_state=random_state
    )

    # Step 2: split remaining into train + val
    val_adjusted = val_size / (1 - test_size)
    strat_temp   = y_temp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_adjusted,
        stratify=strat_temp, random_state=random_state
    )

    total = len(X)
    print(f"\n[Three-Way Split]")
    print(f"  Total   : {total:>6} rows (100%)")
    print(f"  Train   : {len(X_train):>6} rows ({len(X_train)/total*100:.1f}%)")
    print(f"  Val     : {len(X_val):>6} rows ({len(X_val)/total*100:.1f}%)")
    print(f"  Test    : {len(X_test):>6} rows ({len(X_test)/total*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# 🔧 4. K-FOLD CROSS-VALIDATION
# =============================================================================

def kfold_cross_validation(X: pd.DataFrame,
                             y: pd.Series,
                             model,
                             n_splits: int = 5,
                             scoring: str = "accuracy",
                             shuffle: bool = True,
                             random_state: int = 42) -> dict:
    """
    Performs K-Fold Cross-Validation and returns fold-level scores.

    Best for:
        - Small-to-medium datasets
        - More reliable evaluation than a single split
        - Regression or balanced classification

    Args:
        X            : Feature DataFrame or array
        y            : Target Series or array
        model        : Scikit-learn estimator
        n_splits     : Number of folds (default: 5)
        scoring      : Evaluation metric (default: 'accuracy')
        shuffle      : Shuffle data before splitting (default: True)
        random_state : Reproducibility seed (default: 42)

    Returns:
        Dictionary with fold scores, mean, and std
    """
    kf     = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=kf, scoring=scoring)

    result = {
        "fold_scores" : scores.tolist(),
        "mean"        : round(scores.mean(), 4),
        "std"         : round(scores.std(), 4),
        "min"         : round(scores.min(), 4),
        "max"         : round(scores.max(), 4),
    }

    print(f"\n[KFold CV] k={n_splits} | metric={scoring}")
    for i, s in enumerate(scores, 1):
        print(f"  Fold {i}: {s:.4f}")
    print(f"  ──────────────────────")
    print(f"  Mean ± Std : {result['mean']:.4f} ± {result['std']:.4f}")
    print(f"  Min / Max  : {result['min']:.4f} / {result['max']:.4f}")

    return result


# =============================================================================
# 🔧 5. STRATIFIED K-FOLD
# =============================================================================

def stratified_kfold_cv(X: pd.DataFrame,
                          y: pd.Series,
                          model,
                          n_splits: int = 5,
                          scoring: str = "accuracy",
                          random_state: int = 42) -> dict:
    """
    Performs Stratified K-Fold Cross-Validation.

    Preserves class distribution in every fold.

    Best for:
        - Classification tasks — always prefer over plain K-Fold
        - Imbalanced datasets
        - Default CV strategy for classifiers

    Args:
        X            : Feature DataFrame or array
        y            : Target Series (categorical/binary)
        model        : Scikit-learn estimator
        n_splits     : Number of folds (default: 5)
        scoring      : Evaluation metric (default: 'accuracy')
        random_state : Reproducibility seed (default: 42)

    Returns:
        Dictionary with fold scores, mean, and std
    """
    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)

    result = {
        "fold_scores": scores.tolist(),
        "mean"       : round(scores.mean(), 4),
        "std"        : round(scores.std(), 4),
    }

    print(f"\n[Stratified KFold CV] k={n_splits} | metric={scoring}")
    for i, s in enumerate(scores, 1):
        print(f"  Fold {i}: {s:.4f}")
    print(f"  ──────────────────────")
    print(f"  Mean ± Std : {result['mean']:.4f} ± {result['std']:.4f}")

    return result


# =============================================================================
# 🔧 6. TIME SERIES SPLIT
# =============================================================================

def time_series_split_cv(X: pd.DataFrame,
                           y: pd.Series,
                           model,
                           n_splits: int = 5,
                           scoring: str = "accuracy") -> dict:
    """
    Performs Time Series Cross-Validation (expanding window).

    Respects temporal ordering:
        - Training always uses past data only
        - Validation always uses future data
        - No shuffling — order is preserved

    Best for:
        - Time-series data (stock prices, sales, weather)
        - Any dataset with temporal dependencies

    Args:
        X        : Feature DataFrame (sorted by time — earliest first)
        y        : Target Series
        model    : Scikit-learn estimator
        n_splits : Number of expanding splits (default: 5)
        scoring  : Evaluation metric (default: 'accuracy')

    Returns:
        Dictionary with fold scores, mean, and std
    """
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring)

    result = {
        "fold_scores": scores.tolist(),
        "mean"       : round(scores.mean(), 4),
        "std"        : round(scores.std(), 4),
    }

    print(f"\n[TimeSeriesSplit CV] n_splits={n_splits} | metric={scoring}")
    for i, s in enumerate(scores, 1):
        train_end = int(len(X) * (i / (n_splits + 1)))
        print(f"  Fold {i}: {s:.4f}  (train size ≈ {train_end})")
    print(f"  ──────────────────────")
    print(f"  Mean ± Std : {result['mean']:.4f} ± {result['std']:.4f}")

    return result


# =============================================================================
# 🔧 7. GROUP K-FOLD
# =============================================================================

def group_kfold_cv(X: pd.DataFrame,
                    y: pd.Series,
                    groups: pd.Series,
                    model,
                    n_splits: int = 5,
                    scoring: str = "accuracy") -> dict:
    """
    Performs Group K-Fold Cross-Validation.

    Ensures no group (patient, user, location) appears in both
    train and validation splits — prevents group-level leakage.

    Best for:
        - Medical data (same patient in train and test = leakage)
        - User behavior data
        - Any hierarchically grouped data

    Args:
        X        : Feature DataFrame
        y        : Target Series
        groups   : Series of group labels for each row
        model    : Scikit-learn estimator
        n_splits : Number of folds (default: 5)
        scoring  : Evaluation metric (default: 'accuracy')

    Returns:
        Dictionary with fold scores, mean, and std
    """
    gkf    = GroupKFold(n_splits=n_splits)
    scores = cross_val_score(model, X, y, cv=gkf, groups=groups, scoring=scoring)

    result = {
        "fold_scores": scores.tolist(),
        "mean"       : round(scores.mean(), 4),
        "std"        : round(scores.std(), 4),
    }

    print(f"\n[GroupKFold CV] n_splits={n_splits} | groups={groups.nunique()} unique")
    for i, s in enumerate(scores, 1):
        print(f"  Fold {i}: {s:.4f}")
    print(f"  ──────────────────────")
    print(f"  Mean ± Std : {result['mean']:.4f} ± {result['std']:.4f}")

    return result


# =============================================================================
# 🔧 8. LEAVE-ONE-OUT CROSS-VALIDATION (LOOCV)
# =============================================================================

def loocv(X: pd.DataFrame,
           y: pd.Series,
           model,
           scoring: str = "accuracy") -> dict:
    """
    Performs Leave-One-Out Cross-Validation (LOOCV).

    Each sample is used once as the validation set (K = N folds).

    Best for:
        - Very small datasets (< 50 samples)
        - Maximum use of available data

    ⚠️ Warning: Computationally expensive on large datasets (N model fits).

    Args:
        X       : Feature DataFrame
        y       : Target Series
        model   : Scikit-learn estimator
        scoring : Evaluation metric (default: 'accuracy')

    Returns:
        Dictionary with all fold scores, mean, and std
    """
    if len(X) > 500:
        print(f"[LOOCV] ⚠️  Dataset has {len(X)} rows — LOOCV may be slow. "
              f"Consider K-Fold instead.")

    loo    = LeaveOneOut()
    scores = cross_val_score(model, X, y, cv=loo, scoring=scoring)

    result = {
        "n_folds"    : len(scores),
        "mean"       : round(scores.mean(), 4),
        "std"        : round(scores.std(), 4),
        "min"        : round(scores.min(), 4),
        "max"        : round(scores.max(), 4),
    }

    print(f"\n[LOOCV] n_folds={len(scores)} | metric={scoring}")
    print(f"  Mean ± Std : {result['mean']:.4f} ± {result['std']:.4f}")
    print(f"  Min / Max  : {result['min']:.4f} / {result['max']:.4f}")

    return result


# =============================================================================
# 🔧 9. MULTIPLE METRICS CROSS-VALIDATION
# =============================================================================

def cv_multiple_metrics(X: pd.DataFrame,
                          y: pd.Series,
                          model,
                          n_splits: int = 5,
                          metrics: list = None,
                          stratified: bool = True,
                          random_state: int = 42) -> pd.DataFrame:
    """
    Performs cross-validation scoring across multiple metrics simultaneously.

    Args:
        X            : Feature DataFrame
        y            : Target Series
        model        : Scikit-learn estimator
        n_splits     : Number of folds (default: 5)
        metrics      : List of metric names (default: accuracy, f1, roc_auc, precision, recall)
        stratified   : Use stratified folds (default: True)
        random_state : Reproducibility seed (default: 42)

    Returns:
        DataFrame with mean and std for each metric
    """
    if metrics is None:
        metrics = ["accuracy", "f1_weighted", "roc_auc", "precision_weighted", "recall_weighted"]

    cv = (StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
          if stratified else
          KFold(n_splits=n_splits, shuffle=True, random_state=random_state))

    cv_results = cross_validate(model, X, y, cv=cv, scoring=metrics)

    rows = []
    for metric in metrics:
        key    = f"test_{metric}"
        scores = cv_results[key]
        rows.append({
            "Metric"   : metric,
            "Mean"     : round(scores.mean(), 4),
            "Std"      : round(scores.std(), 4),
            "Min"      : round(scores.min(), 4),
            "Max"      : round(scores.max(), 4),
        })

    report = pd.DataFrame(rows)
    print(f"\n[Multi-Metric CV] k={n_splits} | stratified={stratified}")
    print(report.to_string(index=False))
    return report


# =============================================================================
# 🔧 10. LEAKAGE-SAFE PREPROCESSING PIPELINE
# =============================================================================

def build_safe_pipeline(model,
                          scaler=None) -> Pipeline:
    """
    Builds a sklearn Pipeline that applies preprocessing safely inside CV.

    The Pipeline ensures the scaler is fit only on training folds
    and transforms validation/test folds — preventing data leakage.

    Args:
        model  : Scikit-learn estimator
        scaler : Scikit-learn scaler (default: StandardScaler)

    Returns:
        sklearn Pipeline object

    Usage:
        pipe = build_safe_pipeline(LogisticRegression())
        scores = cross_val_score(pipe, X, y, cv=5)
    """
    if scaler is None:
        scaler = StandardScaler()

    pipeline = Pipeline([
        ("scaler", scaler),
        ("model",  model),
    ])
    print(f"[Pipeline] Built: {type(scaler).__name__} → {type(model).__name__}")
    print("  ✅ Scaler fits inside CV — no leakage!")
    return pipeline


# =============================================================================
# 🔧 11. UTILITY — SPLIT VALIDATION REPORT
# =============================================================================

def split_validation_report(X_train: pd.DataFrame,
                               X_test: pd.DataFrame,
                               y_train: pd.Series,
                               y_test: pd.Series) -> None:
    """
    Prints a validation report comparing train and test distributions.

    Args:
        X_train : Training features
        X_test  : Test features
        y_train : Training target
        y_test  : Test target
    """
    print("\n📋 Split Validation Report")
    print("=" * 50)
    total = len(X_train) + len(X_test)
    print(f"  Total rows   : {total}")
    print(f"  Train rows   : {len(X_train)} ({len(X_train)/total*100:.1f}%)")
    print(f"  Test rows    : {len(X_test)} ({len(X_test)/total*100:.1f}%)")

    if y_train is not None:
        print(f"\n  Target Distribution:")
        train_dist = y_train.value_counts(normalize=True).round(3)
        test_dist  = y_test.value_counts(normalize=True).round(3)
        for cls in sorted(train_dist.index):
            print(f"    Class {cls} → Train: {train_dist.get(cls, 0):.3f} | "
                  f"Test: {test_dist.get(cls, 0):.3f}")

    print(f"\n  Feature Summary (numeric):")
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    for col in num_cols[:5]:   # show first 5 numeric features
        print(f"    {col:20s} | Train mean: {X_train[col].mean():.2f} | "
              f"Test mean: {X_test[col].mean():.2f}")
    if len(num_cols) > 5:
        print(f"    ... and {len(num_cols) - 5} more numeric features")
    print("=" * 50)


# =============================================================================
# 🔧 12. UTILITY — INTERNAL SPLIT REPORT HELPER
# =============================================================================

def _split_report(name: str,
                   X: pd.DataFrame,
                   X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   y_train: pd.Series,
                   y_test: pd.Series) -> None:
    """Internal helper to print a concise split summary."""
    total = len(X)
    print(f"\n[{name}]")
    print(f"  Total  : {total} rows")
    print(f"  Train  : {len(X_train)} rows ({len(X_train)/total*100:.1f}%)")
    print(f"  Test   : {len(X_test)} rows ({len(X_test)/total*100:.1f}%)")


# =============================================================================
# 🚀 MAIN — Demo with Synthetic Dataset
# =============================================================================

if __name__ == "__main__":

    # ── Sample Dataset ──────────────────────────────────────────────────────
    np.random.seed(42)
    n = 1000

    data = {
        "Age"       : np.random.randint(20, 65, n),
        "Salary"    : np.random.randint(30_000, 150_000, n),
        "Score"     : np.random.uniform(0, 100, n),
        "Experience": np.random.randint(0, 30, n),
        "Group"     : np.random.choice([f"Patient_{i}" for i in range(50)], n),
    }
    # Imbalanced binary target (80/20)
    target = np.random.choice([0, 1], size=n, p=[0.8, 0.2])

    df      = pd.DataFrame(data)
    X       = df[["Age", "Salary", "Score", "Experience"]]
    y       = pd.Series(target, name="Target")
    groups  = df["Group"]

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    pipe  = build_safe_pipeline(LogisticRegression(max_iter=500), StandardScaler())

    print("=" * 65)
    print("📊 Dataset Overview")
    print("=" * 65)
    print(f"Shape: {X.shape} | Class balance: {dict(pd.Series(target).value_counts())}")

    # ── 1. Simple Random Split ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("1️⃣  Simple Random Split")
    print("=" * 65)
    X_tr, X_te, y_tr, y_te = simple_split(X, y, test_size=0.2)
    split_validation_report(X_tr, X_te, y_tr, y_te)

    # ── 2. Stratified Split ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("2️⃣  Stratified Split")
    print("=" * 65)
    X_tr_s, X_te_s, y_tr_s, y_te_s = stratified_split(X, y, test_size=0.2)

    # ── 3. Three-Way Split ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("3️⃣  Three-Way Split (Train / Val / Test)")
    print("=" * 65)
    X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(
        X, y, val_size=0.15, test_size=0.15
    )

    # ── 4. K-Fold CV ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("4️⃣  K-Fold Cross-Validation (k=5)")
    print("=" * 65)
    kf_results = kfold_cross_validation(X, y, model, n_splits=5)

    # ── 5. Stratified K-Fold ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("5️⃣  Stratified K-Fold CV (k=5)")
    print("=" * 65)
    skf_results = stratified_kfold_cv(X, y, model, n_splits=5)

    # ── 6. Time Series Split ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("6️⃣  Time Series Split CV")
    print("=" * 65)
    ts_results = time_series_split_cv(X, y, model, n_splits=5)

    # ── 7. Group K-Fold ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("7️⃣  Group K-Fold CV")
    print("=" * 65)
    gkf_results = group_kfold_cv(X, y, groups, model, n_splits=5)

    # ── 8. LOOCV (small subset) ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("8️⃣  LOOCV (on 40-sample subset)")
    print("=" * 65)
    loocv_results = loocv(X.iloc[:40], y.iloc[:40], model)

    # ── 9. Multi-Metric CV ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("9️⃣  Multi-Metric Stratified K-Fold")
    print("=" * 65)
    metrics_report = cv_multiple_metrics(
        X, y, model, n_splits=5,
        metrics=["accuracy", "f1_weighted", "roc_auc", "precision_weighted", "recall_weighted"]
    )

    # ── 10. Safe Pipeline CV ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("🔧  Leakage-Safe Pipeline CV")
    print("=" * 65)
    pipe_scores = cross_val_score(pipe, X, y, cv=StratifiedKFold(5), scoring="accuracy")
    print(f"  Pipeline CV Scores: {np.round(pipe_scores, 4)}")
    print(f"  Mean ± Std        : {pipe_scores.mean():.4f} ± {pipe_scores.std():.4f}")

    print("\n✅ All Train-Test Split strategies demonstrated successfully!")
