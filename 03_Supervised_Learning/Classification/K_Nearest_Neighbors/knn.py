# =============================================================================
# 📦 K-Nearest Neighbors (KNN) — Reusable ML Script
# =============================================================================
# Author   : Hamna Munir
# Topic    : 03_Supervised_Learning / Classification / K_Nearest_Neighbors
# File     : knn.py
# =============================================================================

# -----------------------------------------------------------------------------
# 📚 Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold, GridSearchCV, validation_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 🔧 1. TRAIN KNN CLASSIFIER
# =============================================================================

def train_knn_classifier(X_train: pd.DataFrame,
                           X_test: pd.DataFrame,
                           y_train: pd.Series,
                           y_test: pd.Series,
                           n_neighbors: int = 5,
                           weights: str = "uniform",
                           metric: str = "euclidean",
                           algorithm: str = "auto",
                           scale: bool = True) -> dict:
    """
    Trains a K-Nearest Neighbors classifier.

    Algorithm:
        1. Store all training data (no model is built)
        2. For each test point:
           a. Compute distance to all training points
           b. Select K nearest neighbors
           c. Return majority class (or weighted vote)

    Args:
        X_train     : Training features DataFrame
        X_test      : Test features DataFrame
        y_train     : Training target Series
        y_test      : Test target Series
        n_neighbors : Number of neighbors K (default: 5)
        weights     : 'uniform' (equal votes) or 'distance' (weighted by 1/d)
        metric      : Distance metric — 'euclidean', 'manhattan', 'minkowski'
        algorithm   : Search algorithm — 'auto', 'ball_tree', 'kd_tree', 'brute'
        scale       : Whether to StandardScale features (default: True)

    Returns:
        Dictionary with model, predictions, probabilities, and metrics
    """
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        algorithm=algorithm,
        n_jobs=-1
    )))

    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)

    y_pred   = pipe.predict(X_test)
    y_prob   = pipe.predict_proba(X_test)
    n_classes = len(np.unique(y_train))

    metrics  = _evaluate_classifier(y_test, y_pred,
                                     y_prob[:, 1] if n_classes == 2 else None,
                                     n_classes)

    print(f"[KNN Classifier] K={n_neighbors} | weights={weights} | "
          f"metric={metric}")
    _print_clf_metrics(metrics)

    return {
        "pipeline"    : pipe,
        "model"       : pipe.named_steps["model"],
        "y_pred"      : y_pred,
        "y_prob"      : y_prob,
        "metrics"     : metrics,
        "n_neighbors" : n_neighbors,
        "weights"     : weights,
        "metric"      : metric,
    }


# =============================================================================
# 🔧 2. TRAIN KNN REGRESSOR
# =============================================================================

def train_knn_regressor(X_train: pd.DataFrame,
                         X_test: pd.DataFrame,
                         y_train: pd.Series,
                         y_test: pd.Series,
                         n_neighbors: int = 5,
                         weights: str = "uniform",
                         metric: str = "euclidean",
                         scale: bool = True) -> dict:
    """
    Trains a K-Nearest Neighbors regressor.

    Prediction:
        uniform  → ŷ = mean(y of K neighbors)
        distance → ŷ = weighted mean (closer = higher weight = 1/d²)

    Args:
        X_train     : Training features DataFrame
        X_test      : Test features DataFrame
        y_train     : Training target Series (continuous)
        y_test      : Test target Series
        n_neighbors : Number of neighbors K (default: 5)
        weights     : 'uniform' or 'distance'
        metric      : Distance metric
        scale       : Whether to StandardScale (default: True)

    Returns:
        Dictionary with model, predictions, and regression metrics
    """
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        n_jobs=-1
    )))

    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)

    y_pred_train = pipe.predict(X_train)
    y_pred_test  = pipe.predict(X_test)
    metrics      = _evaluate_regressor(y_train, y_pred_train,
                                        y_test, y_pred_test)

    print(f"[KNN Regressor] K={n_neighbors} | weights={weights}")
    print(f"  [TRAIN] RMSE={metrics['train']['RMSE']:.4f} | "
          f"R²={metrics['train']['R²']:.4f}")
    print(f"  [TEST ] RMSE={metrics['test']['RMSE']:.4f} | "
          f"R²={metrics['test']['R²']:.4f}")

    return {
        "pipeline"    : pipe,
        "model"       : pipe.named_steps["model"],
        "y_pred_train": y_pred_train,
        "y_pred_test" : y_pred_test,
        "metrics"     : metrics,
    }


# =============================================================================
# 🔧 3. FIND OPTIMAL K — VALIDATION CURVE
# =============================================================================

def find_optimal_k(X_train: pd.DataFrame,
                    y_train: pd.Series,
                    k_range: range = None,
                    cv: int = 5,
                    scoring: str = "accuracy",
                    task: str = "classification") -> dict:
    """
    Evaluates KNN performance across a range of K values using cross-validation.

    This is the recommended method to select the optimal K:
    - Plots train and validation scores per K
    - Finds the K that maximizes validation score

    Args:
        X_train  : Training features DataFrame
        y_train  : Training target Series
        k_range  : Range of K values to evaluate (default: 1 to 30)
        cv       : Number of CV folds (default: 5)
        scoring  : Scoring metric (default: 'accuracy')
        task     : 'classification' or 'regression'

    Returns:
        Dictionary with optimal K, scores, and validation curve data
    """
    if k_range is None:
        k_range = range(1, 31)

    if task == "classification":
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  KNeighborsClassifier(n_jobs=-1))
        ])
        param_name = "model__n_neighbors"
    else:
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  KNeighborsRegressor(n_jobs=-1))
        ])
        param_name = "model__n_neighbors"
        if scoring == "accuracy":
            scoring = "r2"

    train_scores, val_scores = validation_curve(
        estimator, X_train, y_train,
        param_name=param_name,
        param_range=list(k_range),
        cv=cv, scoring=scoring, n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)
    optimal_k  = list(k_range)[np.argmax(val_mean)]

    print(f"[Optimal K Search] Scoring={scoring} | CV={cv}")
    print(f"  Optimal K    : {optimal_k}")
    print(f"  Val {scoring}: {val_mean[optimal_k-1]:.4f} "
          f"± {val_std[optimal_k-1]:.4f}")

    return {
        "optimal_k"   : optimal_k,
        "k_range"     : list(k_range),
        "train_mean"  : train_mean,
        "train_scores": train_scores,
        "val_mean"    : val_mean,
        "val_std"     : val_std,
        "val_scores"  : val_scores,
        "scoring"     : scoring,
    }


# =============================================================================
# 🔧 4. GRIDSEARCHCV — KNN HYPERPARAMETER TUNING
# =============================================================================

def tune_knn(X_train: pd.DataFrame,
              y_train: pd.Series,
              task: str = "classification",
              cv: int = 5,
              scoring: str = "accuracy") -> dict:
    """
    Tunes KNN hyperparameters using GridSearchCV.

    Searches over:
        - n_neighbors (K): number of neighbors
        - weights: 'uniform' vs 'distance'
        - metric: distance function

    Args:
        X_train : Training features DataFrame
        y_train : Training target Series
        task    : 'classification' or 'regression'
        cv      : Number of CV folds (default: 5)
        scoring : Scoring metric

    Returns:
        Dictionary with best params, score, and GridSearchCV object
    """
    if task == "classification":
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  KNeighborsClassifier(n_jobs=-1))
        ])
        if scoring == "accuracy":
            scoring = "roc_auc" if len(np.unique(y_train)) == 2 else "accuracy"
    else:
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  KNeighborsRegressor(n_jobs=-1))
        ])
        if scoring == "accuracy":
            scoring = "r2"

    param_grid = {
        "model__n_neighbors": [3, 5, 7, 10, 15, 20, 25, 30],
        "model__weights"    : ["uniform", "distance"],
        "model__metric"     : ["euclidean", "manhattan"],
    }

    grid = GridSearchCV(
        estimator, param_grid,
        cv=cv, scoring=scoring,
        n_jobs=-1, refit=True
    )
    grid.fit(X_train, y_train)

    print(f"[GridSearchCV KNN] Best params: {grid.best_params_}")
    print(f"  Best CV {scoring}: {grid.best_score_:.4f}")

    return {
        "grid"        : grid,
        "best_params" : grid.best_params_,
        "best_score"  : grid.best_score_,
        "best_model"  : grid.best_estimator_,
        "cv_results"  : pd.DataFrame(grid.cv_results_),
    }


# =============================================================================
# 🔧 5. DISTANCE METRIC COMPARISON
# =============================================================================

def compare_distance_metrics(X_train: pd.DataFrame,
                               X_test: pd.DataFrame,
                               y_train: pd.Series,
                               y_test: pd.Series,
                               n_neighbors: int = 5,
                               task: str = "classification") -> pd.DataFrame:
    """
    Compares KNN performance across different distance metrics.

    Metrics tested:
        - euclidean (L2)
        - manhattan (L1)
        - chebyshev (L∞)
        - minkowski (p=3)

    Args:
        X_train     : Training features
        X_test      : Test features
        y_train     : Training labels
        y_test      : Test labels
        n_neighbors : Fixed K value for comparison
        task        : 'classification' or 'regression'

    Returns:
        DataFrame with performance per distance metric
    """
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    metrics_list = [
        ("euclidean",  {}),
        ("manhattan",  {}),
        ("chebyshev",  {}),
        ("minkowski",  {"p": 3}),
    ]

    rows = []
    for metric_name, metric_params in metrics_list:
        if task == "classification":
            model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                metric=metric_name,
                metric_params=metric_params if metric_params else None,
                n_jobs=-1
            )
            model.fit(X_tr_sc, y_train)
            y_pred = model.predict(X_te_sc)
            y_prob = model.predict_proba(X_te_sc)[:, 1]
            rows.append({
                "Distance Metric": metric_name,
                "Accuracy"       : round(accuracy_score(y_test, y_pred), 4),
                "F1 Score"       : round(f1_score(y_test, y_pred,
                                                   average="weighted",
                                                   zero_division=0), 4),
                "ROC-AUC"        : round(roc_auc_score(y_test, y_prob)
                                         if len(np.unique(y_test))==2
                                         else float("nan"), 4),
            })
        else:
            model = KNeighborsRegressor(
                n_neighbors=n_neighbors,
                metric=metric_name,
                n_jobs=-1
            )
            model.fit(X_tr_sc, y_train)
            y_pred = model.predict(X_te_sc)
            rows.append({
                "Distance Metric": metric_name,
                "R²"             : round(r2_score(y_test, y_pred), 4),
                "RMSE"           : round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                "MAE"            : round(mean_absolute_error(y_test, y_pred), 4),
            })

    df = pd.DataFrame(rows)
    print(f"Distance Metric Comparison (K={n_neighbors}):")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 6. WEIGHTS COMPARISON (Uniform vs Distance)
# =============================================================================

def compare_weights(X_train: pd.DataFrame,
                     X_test: pd.DataFrame,
                     y_train: pd.Series,
                     y_test: pd.Series,
                     k_range: list = None) -> pd.DataFrame:
    """
    Compares uniform vs distance-weighted KNN across multiple K values.

    Distance-weighted KNN assigns higher influence to closer neighbors:
        weight = 1 / d²  (where d = distance to neighbor)

    Args:
        X_train : Training features
        X_test  : Test features
        y_train : Training labels
        y_test  : Test labels
        k_range : List of K values to compare

    Returns:
        DataFrame with accuracy per K per weighting scheme
    """
    if k_range is None:
        k_range = [1, 3, 5, 7, 10, 15, 20, 25, 30]

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    rows = []
    for K in k_range:
        for weight in ["uniform", "distance"]:
            model = KNeighborsClassifier(
                n_neighbors=K, weights=weight, n_jobs=-1
            )
            model.fit(X_tr_sc, y_train)
            y_pred = model.predict(X_te_sc)
            rows.append({
                "K"       : K,
                "Weights" : weight,
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "F1"      : round(f1_score(y_test, y_pred,
                                           average="weighted",
                                           zero_division=0), 4),
            })

    df = pd.DataFrame(rows)
    print("Uniform vs Distance Weights Comparison:")
    pivot = df.pivot(index="K", columns="Weights", values="Accuracy")
    print(pivot.to_string())
    return df


# =============================================================================
# 🔧 7. K SENSITIVITY ANALYSIS
# =============================================================================

def k_sensitivity(X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   y_train: pd.Series,
                   y_test: pd.Series,
                   k_range: list = None) -> pd.DataFrame:
    """
    Evaluates train and test accuracy for each K value —
    reveals underfitting (large K) and overfitting (small K).

    Args:
        X_train : Training features
        X_test  : Test features
        y_train : Training labels
        y_test  : Test labels
        k_range : List of K values to evaluate

    Returns:
        DataFrame with train/test accuracy per K
    """
    if k_range is None:
        k_range = list(range(1, 31))

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    rows = []
    for K in k_range:
        model = KNeighborsClassifier(n_neighbors=K, n_jobs=-1)
        model.fit(X_tr_sc, y_train)
        tr_acc = accuracy_score(y_train, model.predict(X_tr_sc))
        te_acc = accuracy_score(y_test,  model.predict(X_te_sc))
        rows.append({
            "K"         : K,
            "Train Acc" : round(tr_acc, 4),
            "Test Acc"  : round(te_acc, 4),
            "Gap"       : round(tr_acc - te_acc, 4),
        })

    df = pd.DataFrame(rows)
    best_k = df.loc[df["Test Acc"].idxmax(), "K"]
    print(f"K Sensitivity Analysis:")
    print(df.to_string(index=False))
    print(f"\nOptimal K (best test accuracy): {best_k}")
    return df


# =============================================================================
# 🔧 8. CROSS-VALIDATION
# =============================================================================

def cross_validate_knn(X: pd.DataFrame,
                        y: pd.Series,
                        n_neighbors: int = 5,
                        cv: int = 5,
                        scoring: str = "accuracy",
                        task: str = "classification") -> dict:
    """
    Performs Stratified K-Fold Cross-Validation on KNN Pipeline.

    Args:
        X           : Full feature DataFrame
        y           : Full target Series
        n_neighbors : Number of neighbors K (default: 5)
        cv          : Number of folds (default: 5)
        scoring     : Scoring metric (default: 'accuracy')
        task        : 'classification' or 'regression'

    Returns:
        Dictionary with fold scores, mean, and std
    """
    if task == "classification":
        model   = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    else:
        model   = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
        cv_splitter = cv
        if scoring == "accuracy":
            scoring = "r2"

    pipe   = Pipeline([("scaler", StandardScaler()), ("model", model)])
    scores = cross_val_score(pipe, X, y, cv=cv_splitter, scoring=scoring, n_jobs=-1)

    result = {
        "scores"      : scores,
        "mean"        : scores.mean(),
        "std"         : scores.std(),
        "fold_results": {f"Fold {i+1}": round(s, 4) for i, s in enumerate(scores)},
    }

    print(f"[KNN CV] K={n_neighbors} | cv={cv} | "
          f"{scoring.upper()}: {scores.mean():.4f} ± {scores.std():.4f}")
    for fold, score in result["fold_results"].items():
        print(f"  {fold}: {score}")

    return result


# =============================================================================
# 🔧 9. EVALUATE CLASSIFIER
# =============================================================================

def evaluate_knn_classifier(y_test: pd.Series,
                              y_pred: np.ndarray,
                              y_prob: np.ndarray = None,
                              model_name: str = "KNN") -> pd.DataFrame:
    """
    Computes a full classification evaluation report.

    Args:
        y_test     : True labels
        y_pred     : Predicted labels
        y_prob     : Predicted probabilities (optional)
        model_name : Name for display

    Returns:
        DataFrame with evaluation metrics
    """
    metrics = {
        "Model"    : model_name,
        "Accuracy" : round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred,
                                           average="weighted",
                                           zero_division=0), 4),
        "Recall"   : round(recall_score(y_test, y_pred,
                                        average="weighted",
                                        zero_division=0), 4),
        "F1 Score" : round(f1_score(y_test, y_pred,
                                    average="weighted",
                                    zero_division=0), 4),
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

def _evaluate_classifier(y_test, y_pred, y_prob=None, n_classes=2):
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


def _evaluate_regressor(y_train, y_pred_train, y_test, y_pred_test):
    def reg_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return {
            "MAE" : round(mean_absolute_error(y_true, y_pred), 4),
            "RMSE": round(np.sqrt(mse), 4),
            "R²"  : round(r2_score(y_true, y_pred), 4),
        }
    return {"train": reg_metrics(y_train, y_pred_train),
            "test" : reg_metrics(y_test,  y_pred_test)}


def _print_clf_metrics(metrics: dict) -> None:
    m = metrics.get("test", {})
    print(f"  [TEST]  Acc={m.get('Accuracy',0):.4f} | "
          f"F1={m.get('F1',0):.4f} | "
          f"AUC={m.get('ROC-AUC','N/A')}")


# =============================================================================
# 🚀 MAIN — Demo with Synthetic Dataset
# =============================================================================

if __name__ == "__main__":

    from sklearn.datasets import make_classification, make_regression

    np.random.seed(42)

    # ── Binary classification dataset ─────────────────────────────────────
    X_raw, y_raw = make_classification(
        n_samples=800, n_features=8, n_informative=5,
        n_redundant=1, n_classes=2, weights=[0.7, 0.3],
        random_state=42
    )
    X = pd.DataFrame(X_raw, columns=[f"Feat_{i+1}" for i in range(8)])
    y = pd.Series(y_raw, name="Target")

    print("=" * 65)
    print("📊 Dataset Info — Binary Classification")
    print("=" * 65)
    print(f"Shape : {X.shape} | Classes: {dict(y.value_counts().sort_index())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── 1. KNN Classifier ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("1️⃣  KNN Classifier (K=5, uniform, euclidean)")
    print("=" * 65)
    result = train_knn_classifier(X_train, X_test, y_train, y_test, n_neighbors=5)

    # ── 2. Optimal K Search ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("2️⃣  Optimal K Search — Validation Curve")
    print("=" * 65)
    k_result = find_optimal_k(X_train, y_train, k_range=range(1, 31))

    # ── 3. K Sensitivity Analysis ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("3️⃣  K Sensitivity (Train vs Test Accuracy)")
    print("=" * 65)
    k_sens = k_sensitivity(X_train, X_test, y_train, y_test)

    # ── 4. Distance Metric Comparison ────────────────────────────────────
    print("\n" + "=" * 65)
    print("4️⃣  Distance Metric Comparison (K=5)")
    print("=" * 65)
    metric_df = compare_distance_metrics(X_train, X_test, y_train, y_test)

    # ── 5. Weights Comparison ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("5️⃣  Uniform vs Distance Weights")
    print("=" * 65)
    weights_df = compare_weights(X_train, X_test, y_train, y_test)

    # ── 6. GridSearchCV ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("6️⃣  GridSearchCV — Full KNN Tuning")
    print("=" * 65)
    gs_result = tune_knn(X_train, y_train, task="classification")

    # ── 7. Cross-Validation ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("7️⃣  Stratified 5-Fold CV")
    print("=" * 65)
    cv_result = cross_validate_knn(
        X, y,
        n_neighbors=k_result["optimal_k"],
        cv=5, scoring="roc_auc"
    )

    # ── 8. KNN Regressor ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("8️⃣  KNN Regressor")
    print("=" * 65)
    X_reg_raw, y_reg_raw = make_regression(
        n_samples=500, n_features=6, noise=15, random_state=42
    )
    X_reg = pd.DataFrame(X_reg_raw, columns=[f"F{i+1}" for i in range(6)])
    y_reg = pd.Series(y_reg_raw, name="Target")

    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    reg_result = train_knn_regressor(Xr_tr, Xr_te, yr_tr, yr_te, n_neighbors=7)

    print("\n✅ All KNN techniques demonstrated successfully!")
