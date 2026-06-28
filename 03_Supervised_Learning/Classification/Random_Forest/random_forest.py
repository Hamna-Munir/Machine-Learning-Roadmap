# =============================================================================
# 📦 Random Forest — Reusable ML Script
# =============================================================================
# Author   : Hamna Munir
# Topic    : 03_Supervised_Learning / Classification / Random_Forest
# File     : random_forest.py
# =============================================================================

# -----------------------------------------------------------------------------
# 📚 Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 🔧 1. TRAIN RANDOM FOREST CLASSIFIER
# =============================================================================

def train_random_forest(X_train: pd.DataFrame,
                          X_test: pd.DataFrame,
                          y_train: pd.Series,
                          y_test: pd.Series,
                          n_estimators: int = 200,
                          max_depth=None,
                          max_features: str = "sqrt",
                          min_samples_split: int = 2,
                          min_samples_leaf: int = 1,
                          class_weight=None,
                          oob_score: bool = True,
                          random_state: int = 42) -> dict:
    """
    Trains a Random Forest Classifier.

    Note: Random Forest does NOT require feature scaling — like Decision Trees,
    splits are based on raw feature values regardless of scale.

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        n_estimators       : Number of trees (default: 200)
        max_depth           : Max depth per tree (default: None = unlimited)
        max_features        : Features considered per split ('sqrt', 'log2', None)
        min_samples_split   : Min samples to split a node
        min_samples_leaf    : Min samples required in a leaf
        class_weight        : 'balanced' for imbalanced classes, or None
        oob_score            : Whether to compute out-of-bag score
        random_state        : Reproducibility seed

    Returns:
        Dictionary with model, predictions, probabilities, metrics, and OOB score
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        oob_score=oob_score,
        bootstrap=True,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred    = model.predict(X_test)
    y_prob    = model.predict_proba(X_test)
    n_classes = len(np.unique(y_train))

    metrics = _evaluate(y_test, y_pred,
                         y_prob[:, 1] if n_classes == 2 else None,
                         n_classes)

    importances = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    print(f"[RandomForest] n_estimators={n_estimators} | max_depth={max_depth} | "
          f"max_features={max_features}")
    if oob_score:
        print(f"  OOB Score: {model.oob_score_:.4f}")
    _print_metrics(metrics)

    return {
        "model"      : model,
        "y_pred"     : y_pred,
        "y_prob"     : y_prob,
        "metrics"    : metrics,
        "importances": importances,
        "oob_score"  : model.oob_score_ if oob_score else None,
    }


# =============================================================================
# 🔧 2. TRAIN RANDOM FOREST REGRESSOR
# =============================================================================

def train_random_forest_regressor(X_train: pd.DataFrame,
                                     X_test: pd.DataFrame,
                                     y_train: pd.Series,
                                     y_test: pd.Series,
                                     n_estimators: int = 200,
                                     max_depth=None,
                                     max_features="sqrt",
                                     oob_score: bool = True,
                                     random_state: int = 42) -> dict:
    """
    Trains a Random Forest Regressor.

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        n_estimators  : Number of trees (default: 200)
        max_depth     : Max depth per tree
        max_features  : Features per split (default: 'sqrt'; classic RF reg
                        uses 1.0 / total — adjust based on dataset)
        oob_score     : Whether to compute out-of-bag score
        random_state  : Reproducibility seed

    Returns:
        Dictionary with model, predictions, and regression metrics
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        oob_score=oob_score,
        bootstrap=True,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)
    metrics      = _evaluate_regression(y_train, y_pred_train, y_test, y_pred_test)

    print(f"[RandomForestRegressor] n_estimators={n_estimators} | "
          f"max_depth={max_depth}")
    if oob_score:
        print(f"  OOB R² Score: {model.oob_score_:.4f}")
    print(f"  [TRAIN] RMSE={metrics['train']['RMSE']:.4f} | R²={metrics['train']['R²']:.4f}")
    print(f"  [TEST ] RMSE={metrics['test']['RMSE']:.4f} | R²={metrics['test']['R²']:.4f}")

    return {
        "model"       : model,
        "y_pred_train": y_pred_train,
        "y_pred_test" : y_pred_test,
        "metrics"     : metrics,
        "importances" : pd.Series(model.feature_importances_, index=X_train.columns)
                          .sort_values(ascending=False),
        "oob_score"   : model.oob_score_ if oob_score else None,
    }


# =============================================================================
# 🔧 3. N_ESTIMATORS SENSITIVITY ANALYSIS
# =============================================================================

def n_estimators_sensitivity(X_train: pd.DataFrame,
                                X_test: pd.DataFrame,
                                y_train: pd.Series,
                                y_test: pd.Series,
                                n_range: list = None) -> pd.DataFrame:
    """
    Evaluates Random Forest performance across a range of n_estimators —
    shows the error plateau effect (diminishing returns from more trees).

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        n_range : List of n_estimators values to evaluate

    Returns:
        DataFrame with metrics per n_estimators value
    """
    if n_range is None:
        n_range = [1, 5, 10, 25, 50, 100, 200, 300, 500]

    rows = []
    for n in n_range:
        model = RandomForestClassifier(
            n_estimators=n, max_features="sqrt",
            random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if len(np.unique(y_train))==2 else None

        row = {
            "n_estimators": n,
            "Train Acc"   : round(accuracy_score(y_train, model.predict(X_train)), 4),
            "Test Acc"    : round(accuracy_score(y_test, y_pred), 4),
            "F1"          : round(f1_score(y_test, y_pred,
                                           average="weighted", zero_division=0), 4),
        }
        if y_prob is not None:
            row["ROC-AUC"] = round(roc_auc_score(y_test, y_prob), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    print("n_estimators Sensitivity Analysis:")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 4. PERMUTATION IMPORTANCE
# =============================================================================

def compute_permutation_importance(model,
                                     X_test: pd.DataFrame,
                                     y_test: pd.Series,
                                     n_repeats: int = 10,
                                     scoring: str = "accuracy",
                                     random_state: int = 42) -> pd.DataFrame:
    """
    Computes Permutation Importance — a more reliable feature importance
    metric than Mean Decrease in Impurity (MDI).

    Method:
        1. Measure baseline model performance on test set
        2. Shuffle one feature's values (breaks relationship with target)
        3. Measure performance drop
        4. Repeat for each feature, n_repeats times

    Args:
        model       : Fitted RandomForestClassifier/Regressor
        X_test      : Test features DataFrame
        y_test      : Test target Series
        n_repeats   : Number of shuffles per feature (default: 10)
        scoring     : Scoring metric for importance calculation
        random_state: Reproducibility seed

    Returns:
        DataFrame with permutation importance per feature
    """
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats, scoring=scoring,
        random_state=random_state, n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "Feature"   : X_test.columns,
        "Importance Mean": result.importances_mean,
        "Importance Std" : result.importances_std,
    }).sort_values("Importance Mean", ascending=False).reset_index(drop=True)

    print(f"[Permutation Importance] n_repeats={n_repeats} | scoring={scoring}")
    print(importance_df.round(4).to_string(index=False))
    return importance_df


# =============================================================================
# 🔧 5. MDI vs PERMUTATION IMPORTANCE COMPARISON
# =============================================================================

def compare_importance_methods(model,
                                  X_test: pd.DataFrame,
                                  y_test: pd.Series) -> pd.DataFrame:
    """
    Compares Mean Decrease in Impurity (MDI) vs Permutation Importance.

    MDI is computed on training data and biased toward high-cardinality
    features. Permutation importance is computed on test data and is
    generally more trustworthy.

    Args:
        model  : Fitted RandomForestClassifier
        X_test : Test features DataFrame
        y_test : Test target Series

    Returns:
        DataFrame comparing both importance rankings
    """
    mdi = pd.Series(model.feature_importances_, index=X_test.columns, name="MDI")

    perm_result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    perm = pd.Series(perm_result.importances_mean, index=X_test.columns,
                      name="Permutation")

    comparison = pd.DataFrame({"MDI": mdi, "Permutation": perm})
    comparison["MDI Rank"] = comparison["MDI"].rank(ascending=False).astype(int)
    comparison["Perm Rank"] = comparison["Permutation"].rank(ascending=False).astype(int)
    comparison = comparison.sort_values("MDI", ascending=False)

    print("MDI vs Permutation Importance Comparison:")
    print(comparison.round(4).to_string())
    return comparison


# =============================================================================
# 🔧 6. GRIDSEARCHCV / RANDOMIZEDSEARCHCV — TUNING
# =============================================================================

def tune_random_forest(X_train: pd.DataFrame,
                         y_train: pd.Series,
                         method: str = "random",
                         n_iter: int = 30,
                         cv: int = 5,
                         scoring: str = "f1_weighted") -> dict:
    """
    Tunes Random Forest hyperparameters using GridSearchCV or
    RandomizedSearchCV (recommended for large search spaces).

    Args:
        X_train : Training features DataFrame
        y_train : Training target Series
        method  : 'grid' or 'random' (default: 'random')
        n_iter  : Iterations for RandomizedSearchCV (default: 30)
        cv      : Number of CV folds
        scoring : Scoring metric

    Returns:
        Dictionary with best params, score, and search object
    """
    param_grid = {
        "n_estimators"     : [100, 200, 300, 500],
        "max_depth"        : [None, 10, 20, 30],
        "max_features"     : ["sqrt", "log2"],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf" : [1, 2, 4],
    }

    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

    if method == "grid":
        search = GridSearchCV(
            base_model, param_grid, cv=cv,
            scoring=scoring, n_jobs=-1
        )
    else:
        search = RandomizedSearchCV(
            base_model, param_grid, n_iter=n_iter,
            cv=cv, scoring=scoring, random_state=42, n_jobs=-1
        )

    search.fit(X_train, y_train)

    print(f"[{'GridSearchCV' if method=='grid' else 'RandomizedSearchCV'} RF] "
          f"Best params: {search.best_params_}")
    print(f"  Best CV {scoring}: {search.best_score_:.4f}")

    return {
        "search"      : search,
        "best_params" : search.best_params_,
        "best_score"  : search.best_score_,
        "best_model"  : search.best_estimator_,
    }


# =============================================================================
# 🔧 7. OOB SCORE VS N_ESTIMATORS
# =============================================================================

def oob_score_progression(X_train: pd.DataFrame,
                            y_train: pd.Series,
                            n_range: list = None) -> pd.DataFrame:
    """
    Tracks how the Out-of-Bag (OOB) score changes as n_estimators increases.

    OOB score provides a free, built-in validation estimate without
    needing a separate held-out test set.

    Args:
        X_train : Training features DataFrame
        y_train : Training target Series
        n_range : List of n_estimators values to evaluate

    Returns:
        DataFrame with OOB score per n_estimators
    """
    if n_range is None:
        n_range = [10, 25, 50, 100, 150, 200, 300, 500]

    rows = []
    for n in n_range:
        model = RandomForestClassifier(
            n_estimators=n, oob_score=True, bootstrap=True,
            random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        rows.append({
            "n_estimators": n,
            "OOB Score"   : round(model.oob_score_, 4),
        })

    df = pd.DataFrame(rows)
    print("OOB Score Progression:")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 8. CROSS-VALIDATION
# =============================================================================

def cross_validate_rf(X: pd.DataFrame,
                        y: pd.Series,
                        n_estimators: int = 200,
                        cv: int = 5,
                        scoring: str = "f1_weighted") -> dict:
    """
    Performs Stratified K-Fold Cross-Validation on Random Forest.

    Args:
        X            : Full feature DataFrame
        y            : Full target Series
        n_estimators : Number of trees (default: 200)
        cv           : Number of folds
        scoring      : Scoring metric

    Returns:
        Dictionary with fold scores, mean, and std
    """
    model  = RandomForestClassifier(
        n_estimators=n_estimators, random_state=42, n_jobs=-1
    )
    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)

    result = {
        "scores"      : scores,
        "mean"        : scores.mean(),
        "std"         : scores.std(),
        "fold_results": {f"Fold {i+1}": round(s, 4) for i, s in enumerate(scores)},
    }

    print(f"[RandomForest CV] n_estimators={n_estimators} | cv={cv} | "
          f"{scoring}: {scores.mean():.4f} ± {scores.std():.4f}")
    for fold, score in result["fold_results"].items():
        print(f"  {fold}: {score}")

    return result


# =============================================================================
# 🔧 9. EVALUATION METRICS
# =============================================================================

def evaluate_rf_classifier(y_test: pd.Series,
                              y_pred: np.ndarray,
                              y_prob: np.ndarray = None,
                              model_name: str = "Random Forest") -> pd.DataFrame:
    """
    Computes and displays a full classification evaluation report.
    """
    metrics = {
        "Model"    : model_name,
        "Accuracy" : round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred,
                                           average="weighted", zero_division=0), 4),
        "Recall"   : round(recall_score(y_test, y_pred,
                                        average="weighted", zero_division=0), 4),
        "F1 Score" : round(f1_score(y_test, y_pred,
                                    average="weighted", zero_division=0), 4),
    }
    if y_prob is not None and len(np.unique(y_test)) == 2:
        metrics["ROC-AUC"] = round(roc_auc_score(y_test, y_prob), 4)

    report = pd.DataFrame([metrics])
    print(f"\n📊 Evaluation — {model_name}")
    print(report.to_string(index=False))
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    return report


# =============================================================================
# 🔧 HELPERS
# =============================================================================

def _evaluate(y_test, y_pred, y_prob=None, n_classes=2):
    m = {
        "Accuracy" : round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred,
                                           average="binary" if n_classes==2 else "weighted",
                                           zero_division=0), 4),
        "Recall"   : round(recall_score(y_test, y_pred,
                                        average="binary" if n_classes==2 else "weighted",
                                        zero_division=0), 4),
        "F1"       : round(f1_score(y_test, y_pred,
                                    average="binary" if n_classes==2 else "weighted",
                                    zero_division=0), 4),
    }
    if y_prob is not None and n_classes == 2:
        m["ROC-AUC"] = round(roc_auc_score(y_test, y_prob), 4)
    return {"test": m}


def _evaluate_regression(y_train, y_pred_train, y_test, y_pred_test):
    def metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return {
            "MAE" : round(mean_absolute_error(y_true, y_pred), 4),
            "RMSE": round(np.sqrt(mse), 4),
            "R²"  : round(r2_score(y_true, y_pred), 4),
        }
    return {"train": metrics(y_train, y_pred_train),
            "test" : metrics(y_test,  y_pred_test)}


def _print_metrics(metrics: dict) -> None:
    m = metrics.get("test", {})
    print(f"  [TEST]  Acc={m.get('Accuracy',0):.4f} | "
          f"F1={m.get('F1',0):.4f} | "
          f"AUC={m.get('ROC-AUC','N/A')}")


# =============================================================================
# 🚀 MAIN — Demo with Synthetic Dataset
# =============================================================================

if __name__ == "__main__":

    from sklearn.datasets import make_classification

    np.random.seed(42)
    X_raw, y_raw = make_classification(
        n_samples=1000, n_features=15, n_informative=8,
        n_redundant=3, n_classes=2, weights=[0.7, 0.3],
        random_state=42
    )
    X = pd.DataFrame(X_raw, columns=[f"Feature_{i+1:02d}" for i in range(15)])
    y = pd.Series(y_raw, name="Target")

    print("=" * 65)
    print("📊 Dataset Info")
    print("=" * 65)
    print(f"Shape: {X.shape} | Classes: {dict(y.value_counts().sort_index())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── 1. Train Random Forest ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("1️⃣  Random Forest (n_estimators=200)")
    print("=" * 65)
    result = train_random_forest(X_train, X_test, y_train, y_test)

    # ── 2. Permutation Importance ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("2️⃣  Permutation Importance")
    print("=" * 65)
    perm_imp = compute_permutation_importance(result["model"], X_test, y_test)

    # ── 3. MDI vs Permutation comparison ────────────────────────────────────
    print("\n" + "=" * 65)
    print("3️⃣  MDI vs Permutation Importance")
    print("=" * 65)
    comparison_df = compare_importance_methods(result["model"], X_test, y_test)

    # ── 4. n_estimators sensitivity ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("4️⃣  n_estimators Sensitivity (Error Plateau)")
    print("=" * 65)
    n_est_df = n_estimators_sensitivity(X_train, X_test, y_train, y_test)

    # ── 5. OOB score progression ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("5️⃣  OOB Score Progression")
    print("=" * 65)
    oob_df = oob_score_progression(X_train, y_train)

    # ── 6. RandomizedSearchCV ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("6️⃣  RandomizedSearchCV — Hyperparameter Tuning")
    print("=" * 65)
    search_result = tune_random_forest(X_train, y_train, method="random", n_iter=20)

    # ── 7. Cross-Validation ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("7️⃣  Stratified 5-Fold CV")
    print("=" * 65)
    cv_result = cross_validate_rf(X, y, n_estimators=200)

    print("\n✅ All Random Forest techniques demonstrated successfully!")
