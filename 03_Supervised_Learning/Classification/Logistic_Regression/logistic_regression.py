# =============================================================================
# 📦 Logistic Regression — Reusable ML Script
# =============================================================================
# Author   : Hamna Munir
# Topic    : 03_Supervised_Learning / Classification / Logistic_Regression
# File     : logistic_regression.py
# =============================================================================

# -----------------------------------------------------------------------------
# 📚 Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    KFold, StratifiedKFold, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 🔧 1. TRAIN LOGISTIC REGRESSION (Binary)
# =============================================================================

def train_logistic_regression(X_train: pd.DataFrame,
                                X_test: pd.DataFrame,
                                y_train: pd.Series,
                                y_test: pd.Series,
                                C: float = 1.0,
                                penalty: str = "l2",
                                solver: str = "lbfgs",
                                max_iter: int = 1000,
                                scale: bool = True,
                                class_weight=None) -> dict:
    """
    Trains a Binary Logistic Regression classifier.

    Formula:
        P(y=1|X) = σ(β₀ + β₁x₁ + ... + βₙxₙ)
        σ(z) = 1 / (1 + e⁻ᶻ)

    Args:
        X_train      : Training features DataFrame
        X_test       : Test features DataFrame
        y_train      : Training target Series (binary: 0/1)
        y_test       : Test target Series
        C            : Inverse regularization strength (default: 1.0)
                       Larger C = less regularization
        penalty      : Regularization type — 'l1', 'l2', 'elasticnet', None
        solver       : Optimization algorithm — 'lbfgs', 'liblinear', 'saga'
        max_iter     : Maximum iterations for convergence (default: 1000)
        scale        : Whether to StandardScale features (default: True)
        class_weight : 'balanced' for imbalanced classes, or None

    Returns:
        Dictionary with model, predictions, probabilities, and metrics
    """
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", LogisticRegression(
        C=C, penalty=penalty, solver=solver,
        max_iter=max_iter, class_weight=class_weight,
        random_state=42
    )))

    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)

    model        = pipe.named_steps["model"]
    y_pred       = pipe.predict(X_test)
    y_prob       = pipe.predict_proba(X_test)[:, 1]
    metrics      = _evaluate_binary(y_test, y_pred, y_prob)

    print(f"[LogisticRegression] C={C} | penalty={penalty} | solver={solver}")
    print(f"  Intercept    : {model.intercept_[0]:.4f}")
    _print_metrics(metrics)

    return {
        "pipeline"     : pipe,
        "model"        : model,
        "y_pred"       : y_pred,
        "y_prob"       : y_prob,
        "metrics"      : metrics,
        "coefficients" : pd.Series(model.coef_[0], index=X_train.columns),
        "intercept"    : model.intercept_[0],
    }


# =============================================================================
# 🔧 2. TRAIN MULTICLASS LOGISTIC REGRESSION
# =============================================================================

def train_multiclass_logistic(X_train: pd.DataFrame,
                               X_test: pd.DataFrame,
                               y_train: pd.Series,
                               y_test: pd.Series,
                               C: float = 1.0,
                               multi_class: str = "multinomial",
                               solver: str = "lbfgs",
                               max_iter: int = 1000,
                               scale: bool = True) -> dict:
    """
    Trains a Multiclass Logistic Regression classifier.

    Strategies:
        multi_class='ovr'         → One-vs-Rest (K binary classifiers)
        multi_class='multinomial' → Softmax (single K-class model)

    Args:
        X_train     : Training features DataFrame
        X_test      : Test features DataFrame
        y_train     : Training target (K classes)
        y_test      : Test target
        C           : Inverse regularization strength (default: 1.0)
        multi_class : Strategy — 'ovr' or 'multinomial'
        solver      : Optimization solver
        max_iter    : Max iterations
        scale       : Standardize features (default: True)

    Returns:
        Dictionary with model, predictions, and metrics
    """
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", LogisticRegression(
        C=C, multi_class=multi_class, solver=solver,
        max_iter=max_iter, random_state=42
    )))

    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)

    y_pred   = pipe.predict(X_test)
    y_prob   = pipe.predict_proba(X_test)
    classes  = pipe.named_steps["model"].classes_

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    print(f"[Multiclass LR] strategy={multi_class} | C={C} | classes={list(classes)}")
    print(f"  Accuracy (test) : {acc:.4f}")
    print(f"  F1 Weighted     : {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in classes]))

    return {
        "pipeline"  : pipe,
        "model"     : pipe.named_steps["model"],
        "y_pred"    : y_pred,
        "y_prob"    : y_prob,
        "classes"   : classes,
        "accuracy"  : acc,
        "f1"        : f1,
    }


# =============================================================================
# 🔧 3. HYPERPARAMETER TUNING — GridSearchCV
# =============================================================================

def tune_logistic_regression(X_train: pd.DataFrame,
                               y_train: pd.Series,
                               cv: int = 5,
                               scoring: str = "roc_auc") -> dict:
    """
    Tunes Logistic Regression hyperparameters using GridSearchCV.

    Searches over:
        - C (regularization strength)
        - penalty (L1 vs L2)
        - solver

    Args:
        X_train : Training features DataFrame
        y_train : Training target Series
        cv      : Number of CV folds (default: 5)
        scoring : Scoring metric (default: 'roc_auc')

    Returns:
        Dictionary with best params, score, and fitted GridSearchCV
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(max_iter=1000, random_state=42))
    ])

    param_grid = [
        {
            "model__C"      : [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "model__penalty": ["l2"],
            "model__solver" : ["lbfgs", "newton-cg"],
        },
        {
            "model__C"      : [0.001, 0.01, 0.1, 1.0, 10.0],
            "model__penalty": ["l1"],
            "model__solver" : ["liblinear", "saga"],
        },
    ]

    grid = GridSearchCV(
        pipe, param_grid,
        cv=cv, scoring=scoring,
        n_jobs=-1, refit=True
    )
    grid.fit(X_train, y_train)

    print(f"[GridSearchCV LR] Best params: {grid.best_params_}")
    print(f"  Best CV {scoring}: {grid.best_score_:.4f}")

    return {
        "grid"        : grid,
        "best_params" : grid.best_params_,
        "best_score"  : grid.best_score_,
        "best_model"  : grid.best_estimator_,
        "cv_results"  : pd.DataFrame(grid.cv_results_),
    }


# =============================================================================
# 🔧 4. THRESHOLD TUNING
# =============================================================================

def tune_threshold(y_test: pd.Series,
                    y_prob: np.ndarray,
                    metric: str = "f1") -> dict:
    """
    Finds the optimal classification threshold by evaluating all thresholds
    from the precision-recall curve.

    Default threshold is 0.5 — but the optimal threshold depends on the
    business cost of false positives vs false negatives.

    Args:
        y_test  : True binary labels
        y_prob  : Predicted probabilities for class 1
        metric  : Metric to optimize — 'f1', 'precision', 'recall'

    Returns:
        Dictionary with optimal threshold, metric scores, and full curve data
    """
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (
        precisions[:-1] + recalls[:-1] + 1e-10
    )

    if metric == "f1":
        best_idx = np.argmax(f1_scores)
    elif metric == "precision":
        best_idx = np.argmax(precisions[:-1])
    elif metric == "recall":
        best_idx = np.argmax(recalls[:-1])
    else:
        best_idx = np.argmax(f1_scores)

    best_threshold = thresholds[best_idx]
    y_pred_tuned   = (y_prob >= best_threshold).astype(int)

    result = {
        "best_threshold": round(best_threshold, 4),
        "best_f1"       : round(f1_scores[best_idx], 4),
        "precision_at_threshold": round(precisions[best_idx], 4),
        "recall_at_threshold"   : round(recalls[best_idx], 4),
        "precisions"    : precisions,
        "recalls"       : recalls,
        "thresholds"    : thresholds,
        "f1_scores"     : f1_scores,
        "y_pred_tuned"  : y_pred_tuned,
    }

    print(f"[Threshold Tuning] Optimizing for: {metric}")
    print(f"  Default (0.5): F1={f1_score(y_test, (y_prob>=0.5).astype(int)):.4f} | "
          f"P={precision_score(y_test, (y_prob>=0.5).astype(int)):.4f} | "
          f"R={recall_score(y_test, (y_prob>=0.5).astype(int)):.4f}")
    print(f"  Optimal ({best_threshold:.4f}): F1={f1_scores[best_idx]:.4f} | "
          f"P={precisions[best_idx]:.4f} | R={recalls[best_idx]:.4f}")

    return result


# =============================================================================
# 🔧 5. CROSS-VALIDATION
# =============================================================================

def cross_validate_logistic(X: pd.DataFrame,
                              y: pd.Series,
                              C: float = 1.0,
                              cv: int = 5,
                              scoring: str = "roc_auc") -> dict:
    """
    Performs Stratified K-Fold Cross-Validation on Logistic Regression.

    Uses StratifiedKFold to preserve class distribution in every fold —
    critical for imbalanced classification tasks.

    Args:
        X       : Full feature DataFrame
        y       : Full target Series
        C       : Regularization parameter (default: 1.0)
        cv      : Number of folds (default: 5)
        scoring : Scoring metric (default: 'roc_auc')

    Returns:
        Dictionary with fold scores, mean, and std
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(C=C, max_iter=1000, random_state=42))
    ])
    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=skf, scoring=scoring)

    result = {
        "scores"      : scores,
        "mean"        : scores.mean(),
        "std"         : scores.std(),
        "fold_results": {f"Fold {i+1}": round(s, 4) for i, s in enumerate(scores)},
    }

    print(f"[LR Stratified CV] C={C} | K={cv} | "
          f"{scoring.upper()}: {scores.mean():.4f} ± {scores.std():.4f}")
    for fold, score in result["fold_results"].items():
        print(f"  {fold}: {score}")

    return result


# =============================================================================
# 🔧 6. EVALUATE BINARY CLASSIFIER
# =============================================================================

def evaluate_classifier(y_test: pd.Series,
                          y_pred: np.ndarray,
                          y_prob: np.ndarray = None,
                          model_name: str = "Logistic Regression") -> pd.DataFrame:
    """
    Computes and displays a full classification evaluation report.

    Metrics:
        Accuracy, Precision, Recall, F1, ROC-AUC, Log Loss

    Args:
        y_test     : True labels
        y_pred     : Predicted labels
        y_prob     : Predicted probabilities (optional, for AUC/LogLoss)
        model_name : Name for display

    Returns:
        DataFrame with all evaluation metrics
    """
    metrics = {
        "Model"    : model_name,
        "Accuracy" : round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall"   : round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1 Score" : round(f1_score(y_test, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        metrics["ROC-AUC"] = round(roc_auc_score(y_test, y_prob), 4)
        metrics["Log Loss"] = round(log_loss(y_test, y_prob), 4)

    report = pd.DataFrame([metrics])
    print(f"\n📊 Evaluation Report — {model_name}")
    print(report.to_string(index=False))
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    return report


# =============================================================================
# 🔧 7. COEFFICIENT SUMMARY
# =============================================================================

def coefficient_summary(model: LogisticRegression,
                          feature_names: list) -> pd.DataFrame:
    """
    Builds a coefficient table with odds ratios for interpretation.

    Odds Ratio = exp(coefficient):
        OR > 1 → Feature increases odds of class 1
        OR < 1 → Feature decreases odds of class 1
        OR = 1 → Feature has no effect

    Args:
        model        : Fitted LogisticRegression model
        feature_names: List of feature column names

    Returns:
        DataFrame with coefficients and odds ratios
    """
    coefs = model.coef_[0]
    summary = pd.DataFrame({
        "Feature"     : feature_names,
        "Coefficient" : coefs.round(4),
        "Odds Ratio"  : np.exp(coefs).round(4),
        "|Coefficient|": np.abs(coefs).round(4),
        "Direction"   : ["Positive ↑" if c > 0 else "Negative ↓" for c in coefs],
    }).sort_values("|Coefficient|", ascending=False).reset_index(drop=True)

    print("\n[Coefficient Summary]")
    print(summary.to_string(index=False))
    return summary


# =============================================================================
# 🔧 8. C SENSITIVITY ANALYSIS
# =============================================================================

def c_sensitivity(X_train: pd.DataFrame,
                   X_test: pd.DataFrame,
                   y_train: pd.Series,
                   y_test: pd.Series,
                   C_values: list = None) -> pd.DataFrame:
    """
    Evaluates Logistic Regression performance across a range of C values.

    C = 1/lambda (inverse regularization):
        Small C → strong regularization → simpler model
        Large C → weak regularization → complex model

    Args:
        X_train  : Training features
        X_test   : Test features
        y_train  : Training labels
        y_test   : Test labels
        C_values : List of C values to evaluate

    Returns:
        DataFrame with metrics per C value
    """
    if C_values is None:
        C_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    rows = []
    for C in C_values:
        model = LogisticRegression(C=C, max_iter=1000, random_state=42)
        model.fit(X_tr_sc, y_train)
        y_tr_p = model.predict(X_tr_sc)
        y_te_p = model.predict(X_te_sc)
        y_prob = model.predict_proba(X_te_sc)[:, 1]

        rows.append({
            "C"            : C,
            "Train Acc"    : round(accuracy_score(y_train, y_tr_p), 4),
            "Test Acc"     : round(accuracy_score(y_test,  y_te_p), 4),
            "Test F1"      : round(f1_score(y_test, y_te_p, zero_division=0), 4),
            "Test AUC"     : round(roc_auc_score(y_test, y_prob), 4),
            "Coef L2 Norm" : round(np.sqrt(np.sum(model.coef_**2)), 4),
        })

    df = pd.DataFrame(rows)
    print("C Sensitivity Analysis:")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 HELPERS
# =============================================================================

def _evaluate_binary(y_test, y_pred, y_prob=None):
    metrics = {
        "train": {},
        "test" : {
            "Accuracy" : round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall"   : round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1"       : round(f1_score(y_test, y_pred, zero_division=0), 4),
        }
    }
    if y_prob is not None:
        metrics["test"]["ROC-AUC"] = round(roc_auc_score(y_test, y_prob), 4)
        metrics["test"]["Log Loss"] = round(log_loss(y_test, y_prob), 4)
    return metrics


def _print_metrics(metrics: dict) -> None:
    m = metrics.get("test", {})
    print(f"  [TEST]  Acc={m.get('Accuracy',0):.4f} | "
          f"F1={m.get('F1',0):.4f} | "
          f"AUC={m.get('ROC-AUC','N/A')} | "
          f"LogLoss={m.get('Log Loss','N/A')}")


# =============================================================================
# 🚀 MAIN — Demo with Synthetic Dataset
# =============================================================================

if __name__ == "__main__":

    from sklearn.datasets import make_classification

    # ── Synthetic binary classification dataset ────────────────────────────
    np.random.seed(42)
    X_raw, y_raw = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        weights=[0.75, 0.25],   # slightly imbalanced
        random_state=42
    )

    X = pd.DataFrame(X_raw, columns=[f"Feature_{i+1}" for i in range(10)])
    y = pd.Series(y_raw, name="Target")

    print("=" * 65)
    print("📊 Dataset Info — Binary Classification")
    print("=" * 65)
    print(f"Shape    : {X.shape}")
    print(f"Class dist: {dict(y.value_counts().sort_index())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── 1. Binary Logistic Regression ─────────────────────────────────────
    print("\n" + "=" * 65)
    print("1️⃣  Binary Logistic Regression (C=1.0, L2)")
    print("=" * 65)
    result = train_logistic_regression(X_train, X_test, y_train, y_test)

    # ── 2. Coefficient Summary ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("2️⃣  Coefficient Summary + Odds Ratios")
    print("=" * 65)
    coef_df = coefficient_summary(result["model"], X_train.columns.tolist())

    # ── 3. Full Evaluation ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("3️⃣  Full Classification Report")
    print("=" * 65)
    eval_df = evaluate_classifier(
        y_test, result["y_pred"], result["y_prob"]
    )

    # ── 4. Threshold Tuning ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("4️⃣  Threshold Tuning (optimize F1)")
    print("=" * 65)
    thresh_result = tune_threshold(y_test, result["y_prob"], metric="f1")

    # ── 5. C Sensitivity ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("5️⃣  C Sensitivity Analysis")
    print("=" * 65)
    c_df = c_sensitivity(X_train, X_test, y_train, y_test)

    # ── 6. GridSearchCV ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("6️⃣  GridSearchCV — Hyperparameter Tuning")
    print("=" * 65)
    gs_result = tune_logistic_regression(X_train, y_train)

    # ── 7. Stratified CV ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("7️⃣  Stratified K-Fold CV")
    print("=" * 65)
    cv_result = cross_validate_logistic(X, y, C=1.0, cv=5)

    # ── 8. Class weights for imbalance ───────────────────────────────────
    print("\n" + "=" * 65)
    print("8️⃣  Handling Class Imbalance (class_weight='balanced')")
    print("=" * 65)
    result_bal = train_logistic_regression(
        X_train, X_test, y_train, y_test,
        class_weight="balanced"
    )

    print("\n✅ All Logistic Regression techniques demonstrated successfully!")
