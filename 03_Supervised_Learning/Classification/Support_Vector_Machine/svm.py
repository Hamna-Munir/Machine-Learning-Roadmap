# =============================================================================
# 📦 Support Vector Machine (SVM) — Reusable ML Script
# =============================================================================
# Author   : Hamna Munir
# Topic    : 03_Supervised_Learning / Classification / Support_Vector_Machine
# File     : svm.py
# =============================================================================

# -----------------------------------------------------------------------------
# 📚 Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 🔧 1. TRAIN SVM CLASSIFIER
# =============================================================================

def train_svm(X_train: pd.DataFrame,
               X_test: pd.DataFrame,
               y_train: pd.Series,
               y_test: pd.Series,
               kernel: str = "rbf",
               C: float = 1.0,
               gamma: str = "scale",
               degree: int = 3,
               probability: bool = True,
               class_weight=None,
               random_state: int = 42) -> dict:
    """
    Trains a Support Vector Machine classifier.

    ⚠️ ALWAYS scale features before SVM — it is extremely scale-sensitive.

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        kernel       : Kernel type — 'rbf', 'linear', 'poly', 'sigmoid'
        C            : Regularization (larger C = less regularization)
        gamma        : Kernel coefficient — 'scale', 'auto', or float
                       Only used for 'rbf', 'poly', 'sigmoid' kernels
        degree       : Degree for polynomial kernel (ignored for other kernels)
        probability  : Whether to enable probability estimates (slower)
        class_weight : 'balanced' for imbalanced classes, or None
        random_state : Reproducibility seed

    Returns:
        Dictionary with model, predictions, probabilities, and metrics
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  SVC(
            kernel=kernel, C=C, gamma=gamma, degree=degree,
            probability=probability, class_weight=class_weight,
            random_state=random_state
        ))
    ])
    pipe.fit(X_train, y_train)

    model     = pipe.named_steps["model"]
    y_pred    = pipe.predict(X_test)
    n_classes = len(np.unique(y_train))

    y_prob = pipe.predict_proba(X_test) if probability else None

    metrics = _evaluate(
        y_test, y_pred,
        y_prob[:, 1] if (probability and n_classes == 2) else None,
        n_classes
    )

    # Decision function (distance from hyperplane)
    y_decision = pipe.decision_function(X_test)

    print(f"[SVM] kernel={kernel} | C={C} | gamma={gamma}")
    print(f"  n_support_vectors : {model.n_support_.sum()} "
          f"(per class: {model.n_support_})")
    _print_metrics(metrics)

    return {
        "pipeline"     : pipe,
        "model"        : model,
        "y_pred"       : y_pred,
        "y_prob"       : y_prob,
        "y_decision"   : y_decision,
        "metrics"      : metrics,
        "n_support"    : model.n_support_,
    }


# =============================================================================
# 🔧 2. TRAIN LINEAR SVM (Faster for Large Datasets)
# =============================================================================

def train_linear_svm(X_train: pd.DataFrame,
                      X_test: pd.DataFrame,
                      y_train: pd.Series,
                      y_test: pd.Series,
                      C: float = 1.0,
                      max_iter: int = 2000,
                      class_weight=None) -> dict:
    """
    Trains a LinearSVC — much faster than SVC(kernel='linear') for large n.

    Uses liblinear instead of libsvm, which scales to very large datasets.
    Note: LinearSVC does NOT support predict_proba() natively.

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        C            : Regularization strength
        max_iter     : Maximum number of iterations
        class_weight : 'balanced' or None

    Returns:
        Dictionary with model, predictions, and metrics
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearSVC(
            C=C, max_iter=max_iter,
            class_weight=class_weight,
            random_state=42
        ))
    ])
    pipe.fit(X_train, y_train)

    model      = pipe.named_steps["model"]
    y_pred     = pipe.predict(X_test)
    y_decision = pipe.decision_function(X_test)
    n_classes  = len(np.unique(y_train))

    metrics = _evaluate(y_test, y_pred, None, n_classes)

    coefs = None
    if n_classes == 2:
        coefs = pd.Series(
            model.coef_[0], index=X_train.columns
        ).sort_values(key=abs, ascending=False)

    print(f"[LinearSVC] C={C} | max_iter={max_iter}")
    _print_metrics(metrics)

    return {
        "pipeline"   : pipe,
        "model"      : model,
        "y_pred"     : y_pred,
        "y_decision" : y_decision,
        "metrics"    : metrics,
        "coefs"      : coefs,
    }


# =============================================================================
# 🔧 3. TRAIN SVR (Support Vector Regression)
# =============================================================================

def train_svr(X_train: pd.DataFrame,
               X_test: pd.DataFrame,
               y_train: pd.Series,
               y_test: pd.Series,
               kernel: str = "rbf",
               C: float = 1.0,
               epsilon: float = 0.1,
               gamma: str = "scale") -> dict:
    """
    Trains a Support Vector Regressor.

    Loss: Only penalize errors OUTSIDE the ε-tube.
        ε-tube: predictions within ε of true value incur zero loss.

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        kernel  : 'rbf', 'linear', 'poly'
        C       : Regularization strength
        epsilon : Width of ε-insensitive tube
        gamma   : Kernel coefficient

    Returns:
        Dictionary with model, predictions, and regression metrics
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma))
    ])
    pipe.fit(X_train, y_train)

    y_pred_train = pipe.predict(X_train)
    y_pred_test  = pipe.predict(X_test)
    metrics      = _evaluate_regression(y_train, y_pred_train,
                                         y_test, y_pred_test)

    print(f"[SVR] kernel={kernel} | C={C} | epsilon={epsilon}")
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
# 🔧 4. KERNEL COMPARISON
# =============================================================================

def compare_kernels(X_train: pd.DataFrame,
                     X_test: pd.DataFrame,
                     y_train: pd.Series,
                     y_test: pd.Series,
                     C: float = 1.0) -> pd.DataFrame:
    """
    Compares SVM performance across different kernels.

    Kernels tested:
        - linear  : Linear decision boundary
        - rbf     : Gaussian radial basis function (most common)
        - poly(2) : Degree-2 polynomial boundary
        - poly(3) : Degree-3 polynomial boundary
        - sigmoid : Hyperbolic tangent boundary

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        C : Fixed regularization for fair comparison

    Returns:
        DataFrame with metrics per kernel
    """
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    n_classes = len(np.unique(y_train))

    kernel_configs = [
        ("linear",   {"kernel": "linear",  "C": C}),
        ("rbf",      {"kernel": "rbf",     "C": C, "gamma": "scale"}),
        ("poly(2)",  {"kernel": "poly",    "C": C, "degree": 2}),
        ("poly(3)",  {"kernel": "poly",    "C": C, "degree": 3}),
        ("sigmoid",  {"kernel": "sigmoid", "C": C, "gamma": "scale"}),
    ]

    rows = []
    for name, params in kernel_configs:
        model = SVC(**params, probability=True, random_state=42)
        model.fit(X_tr_sc, y_train)
        y_pred = model.predict(X_te_sc)
        y_prob = model.predict_proba(X_te_sc)

        row = {
            "Kernel"      : name,
            "Accuracy"    : round(accuracy_score(y_test, y_pred), 4),
            "F1"          : round(f1_score(y_test, y_pred,
                                           average="weighted",
                                           zero_division=0), 4),
            "N Supports"  : model.n_support_.sum(),
        }
        if n_classes == 2:
            row["ROC-AUC"] = round(roc_auc_score(y_test, y_prob[:, 1]), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Kernel Comparison (C={C}):")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 5. C PARAMETER SENSITIVITY
# =============================================================================

def c_sensitivity(X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   y_train: pd.Series,
                   y_test: pd.Series,
                   kernel: str = "rbf",
                   C_values: list = None) -> pd.DataFrame:
    """
    Evaluates SVM performance across a range of C values.

    C controls the trade-off between maximizing margin and
    minimizing classification errors.

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        kernel   : Fixed kernel for comparison
        C_values : List of C values to evaluate

    Returns:
        DataFrame with train/test metrics per C value
    """
    if C_values is None:
        C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    n_classes = len(np.unique(y_train))

    rows = []
    for C in C_values:
        model = SVC(kernel=kernel, C=C, gamma="scale",
                     probability=True, random_state=42)
        model.fit(X_tr_sc, y_train)
        y_tr_p = model.predict(X_tr_sc)
        y_te_p = model.predict(X_te_sc)
        y_te_prob = model.predict_proba(X_te_sc)

        row = {
            "C"          : C,
            "Train Acc"  : round(accuracy_score(y_train, y_tr_p), 4),
            "Test Acc"   : round(accuracy_score(y_test, y_te_p), 4),
            "Test F1"    : round(f1_score(y_test, y_te_p,
                                          average="weighted",
                                          zero_division=0), 4),
            "N Supports" : model.n_support_.sum(),
        }
        if n_classes == 2:
            row["ROC-AUC"] = round(roc_auc_score(y_test, y_te_prob[:, 1]), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"C Sensitivity ({kernel} kernel):")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 6. GAMMA SENSITIVITY (RBF Kernel)
# =============================================================================

def gamma_sensitivity(X_train: pd.DataFrame,
                       X_test: pd.DataFrame,
                       y_train: pd.Series,
                       y_test: pd.Series,
                       C: float = 1.0,
                       gamma_values: list = None) -> pd.DataFrame:
    """
    Evaluates SVM performance across a range of gamma values (RBF kernel).

    Gamma controls the influence radius of each training point:
        Small gamma → smooth boundary (large influence)
        Large gamma → complex boundary (small influence, risk of overfitting)

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        C             : Fixed C for comparison
        gamma_values  : List of gamma values to evaluate

    Returns:
        DataFrame with train/test metrics per gamma value
    """
    if gamma_values is None:
        gamma_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    n_classes = len(np.unique(y_train))

    rows = []
    for g in gamma_values:
        model = SVC(kernel="rbf", C=C, gamma=g,
                     probability=True, random_state=42)
        model.fit(X_tr_sc, y_train)
        y_tr_p    = model.predict(X_tr_sc)
        y_te_p    = model.predict(X_te_sc)
        y_te_prob = model.predict_proba(X_te_sc)

        row = {
            "Gamma"      : g,
            "Train Acc"  : round(accuracy_score(y_train, y_tr_p), 4),
            "Test Acc"   : round(accuracy_score(y_test, y_te_p), 4),
            "Test F1"    : round(f1_score(y_test, y_te_p,
                                          average="weighted",
                                          zero_division=0), 4),
            "N Supports" : model.n_support_.sum(),
        }
        if n_classes == 2:
            row["ROC-AUC"] = round(roc_auc_score(y_test, y_te_prob[:, 1]), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Gamma Sensitivity (RBF, C={C}):")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 7. GRIDSEARCHCV — SVM TUNING
# =============================================================================

def tune_svm(X_train: pd.DataFrame,
              y_train: pd.Series,
              cv: int = 5,
              scoring: str = "roc_auc") -> dict:
    """
    Tunes SVM hyperparameters using GridSearchCV over C, kernel, and gamma.

    Args:
        X_train : Training features DataFrame
        y_train : Training target Series
        cv      : Number of CV folds (default: 5)
        scoring : Scoring metric (default: 'roc_auc')

    Returns:
        Dictionary with best params, score, and GridSearchCV object
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  SVC(probability=True, random_state=42))
    ])

    param_grid = [
        {
            "model__kernel": ["linear"],
            "model__C"     : [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        {
            "model__kernel": ["rbf"],
            "model__C"     : [0.1, 1.0, 10.0, 100.0],
            "model__gamma" : ["scale", "auto", 0.001, 0.01, 0.1],
        },
        {
            "model__kernel": ["poly"],
            "model__C"     : [0.1, 1.0, 10.0],
            "model__degree": [2, 3],
            "model__gamma" : ["scale"],
        },
    ]

    grid = GridSearchCV(
        pipe, param_grid,
        cv=cv, scoring=scoring,
        n_jobs=-1, refit=True
    )
    grid.fit(X_train, y_train)

    print(f"[GridSearchCV SVM] Best params: {grid.best_params_}")
    print(f"  Best CV {scoring}: {grid.best_score_:.4f}")

    return {
        "grid"       : grid,
        "best_params": grid.best_params_,
        "best_score" : grid.best_score_,
        "best_model" : grid.best_estimator_,
    }


# =============================================================================
# 🔧 8. CROSS-VALIDATION
# =============================================================================

def cross_validate_svm(X: pd.DataFrame,
                         y: pd.Series,
                         kernel: str = "rbf",
                         C: float = 1.0,
                         gamma: str = "scale",
                         cv: int = 5,
                         scoring: str = "roc_auc") -> dict:
    """
    Performs Stratified K-Fold Cross-Validation on SVM Pipeline.

    Args:
        X       : Full feature DataFrame
        y       : Full target Series
        kernel  : SVM kernel type
        C       : Regularization strength
        gamma   : Kernel coefficient
        cv      : Number of folds
        scoring : Scoring metric

    Returns:
        Dictionary with fold scores, mean, and std
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  SVC(kernel=kernel, C=C, gamma=gamma,
                        probability=True, random_state=42))
    ])
    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=skf, scoring=scoring, n_jobs=-1)

    result = {
        "scores"      : scores,
        "mean"        : scores.mean(),
        "std"         : scores.std(),
        "fold_results": {f"Fold {i+1}": round(s, 4) for i, s in enumerate(scores)},
    }

    print(f"[SVM CV] kernel={kernel} | C={C} | cv={cv} | "
          f"{scoring}: {scores.mean():.4f} ± {scores.std():.4f}")
    for fold, score in result["fold_results"].items():
        print(f"  {fold}: {score}")

    return result


# =============================================================================
# 🔧 9. FEATURE IMPORTANCE (Linear SVM Only)
# =============================================================================

def linear_svm_importance(X_train: pd.DataFrame,
                            X_test: pd.DataFrame,
                            y_train: pd.Series,
                            y_test: pd.Series,
                            C: float = 1.0) -> pd.DataFrame:
    """
    Extracts feature importance from a linear SVM via coefficient magnitudes.

    Note: Only linear kernel SVM has interpretable feature weights.
          For non-linear kernels, use permutation importance instead.

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        C : Regularization strength

    Returns:
        DataFrame with feature weights and importance ranking
    """
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    model = LinearSVC(C=C, max_iter=5000, random_state=42)
    model.fit(X_tr_sc, y_train)

    n_classes = len(np.unique(y_train))
    if n_classes == 2:
        coefs = model.coef_[0]
    else:
        coefs = np.mean(np.abs(model.coef_), axis=0)

    importance_df = pd.DataFrame({
        "Feature"       : X_train.columns,
        "Coefficient"   : coefs,
        "|Coefficient|" : np.abs(coefs),
        "Direction"     : ["Positive ↑" if c > 0 else "Negative ↓" for c in coefs],
    }).sort_values("|Coefficient|", ascending=False).reset_index(drop=True)

    y_pred   = model.predict(X_te_sc)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"[Linear SVM Feature Importance] C={C}")
    print(f"  Test Accuracy: {test_acc:.4f}  |  Test F1: {test_f1:.4f}")
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).round(4).to_string(index=False))

    return importance_df


# =============================================================================
# 🔧 10. EVALUATION
# =============================================================================

def evaluate_svm_classifier(y_test: pd.Series,
                              y_pred: np.ndarray,
                              y_prob: np.ndarray = None,
                              model_name: str = "SVM") -> pd.DataFrame:
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

    from sklearn.datasets import make_classification, make_regression

    np.random.seed(42)

    # ── Binary classification dataset ─────────────────────────────────────
    X_raw, y_raw = make_classification(
        n_samples=600, n_features=10, n_informative=6,
        n_redundant=2, n_classes=2, weights=[0.65, 0.35],
        random_state=42
    )
    X = pd.DataFrame(X_raw, columns=[f"Feature_{i+1:02d}" for i in range(10)])
    y = pd.Series(y_raw, name="Target")

    print("=" * 65)
    print("📊 Dataset Info — Binary Classification")
    print("=" * 65)
    print(f"Shape: {X.shape} | Classes: {dict(y.value_counts().sort_index())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── 1. SVM with RBF kernel ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("1️⃣  SVM — RBF Kernel (C=1.0)")
    print("=" * 65)
    result = train_svm(X_train, X_test, y_train, y_test,
                        kernel="rbf", C=1.0)
    evaluate_svm_classifier(y_test, result["y_pred"],
                              result["y_prob"][:, 1] if result["y_prob"] is not None else None)

    # ── 2. Kernel Comparison ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("2️⃣  Kernel Comparison")
    print("=" * 65)
    kernel_df = compare_kernels(X_train, X_test, y_train, y_test, C=1.0)

    # ── 3. C Sensitivity ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("3️⃣  C Parameter Sensitivity (RBF)")
    print("=" * 65)
    c_df = c_sensitivity(X_train, X_test, y_train, y_test, kernel="rbf")

    # ── 4. Gamma Sensitivity ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("4️⃣  Gamma Sensitivity (RBF, C=1.0)")
    print("=" * 65)
    gamma_df = gamma_sensitivity(X_train, X_test, y_train, y_test, C=1.0)

    # ── 5. Linear SVM Feature Importance ──────────────────────────────────
    print("\n" + "=" * 65)
    print("5️⃣  Linear SVM — Feature Importance")
    print("=" * 65)
    imp_df = linear_svm_importance(X_train, X_test, y_train, y_test, C=1.0)

    # ── 6. GridSearchCV ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("6️⃣  GridSearchCV — SVM Tuning")
    print("=" * 65)
    gs_result = tune_svm(X_train, y_train, scoring="roc_auc")

    # ── 7. Cross-Validation ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("7️⃣  Stratified 5-Fold CV")
    print("=" * 65)
    cv_result = cross_validate_svm(X, y, kernel="rbf", C=1.0)

    # ── 8. SVR Demo ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("8️⃣  Support Vector Regression (SVR)")
    print("=" * 65)
    X_reg_raw, y_reg_raw = make_regression(
        n_samples=400, n_features=6, noise=20, random_state=42
    )
    X_reg = pd.DataFrame(X_reg_raw, columns=[f"F{i+1}" for i in range(6)])
    y_reg = pd.Series(y_reg_raw, name="Target")
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    svr_result = train_svr(Xr_tr, Xr_te, yr_tr, yr_te,
                            kernel="rbf", C=1.0, epsilon=0.1)

    print("\n✅ All SVM techniques demonstrated successfully!")
