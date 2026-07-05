# =============================================================================
# 📦 Gradient Boosting — Reusable ML Script
# =============================================================================
# Author   : Hamna Munir
# Topic    : 03_Supervised_Learning / Classification / Gradient_Boosting
# File     : gradient_boosting.py
# =============================================================================

# -----------------------------------------------------------------------------
# 📚 Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.inspection import permutation_importance

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 🔧 1. TRAIN GRADIENT BOOSTING CLASSIFIER
# =============================================================================

def train_gradient_boosting(X_train: pd.DataFrame,
                               X_test: pd.DataFrame,
                               y_train: pd.Series,
                               y_test: pd.Series,
                               n_estimators: int = 200,
                               learning_rate: float = 0.1,
                               max_depth: int = 3,
                               subsample: float = 0.8,
                               max_features: str = "sqrt",
                               min_samples_leaf: int = 1,
                               loss: str = "log_loss",
                               random_state: int = 42) -> dict:
    """
    Trains a Gradient Boosting Classifier (sklearn GBDT).

    Note: GBM does NOT require feature scaling — tree-based splits
    are scale-invariant.

    Algorithm (sequential):
        F₀(x) = constant (log-odds of majority class)
        For m = 1..M:
            rᵢₘ = −∂L/∂F  (pseudo-residuals / negative gradient)
            Fit tree hₘ to {rᵢₘ}
            Fₘ = Fₘ₋₁ + η × hₘ

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        n_estimators   : Number of boosting rounds (trees)
        learning_rate  : Shrinkage — scales each tree contribution (η)
        max_depth      : Max depth per tree (weak learner, keep 3–5)
        subsample      : Fraction of samples per tree (stochastic GBM)
        max_features   : Features per split — 'sqrt', 'log2', None
        min_samples_leaf: Min samples per leaf node
        loss           : Loss function — 'log_loss', 'exponential'
        random_state   : Reproducibility seed

    Returns:
        Dictionary with model, predictions, probabilities, and metrics
    """
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        loss=loss,
        random_state=random_state,
        validation_fraction=0.1,
        n_iter_no_change=20,
        tol=1e-4,
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

    actual_trees = model.n_estimators_
    print(f"[GradientBoosting] n_estimators={n_estimators} "
          f"(actual={actual_trees}) | lr={learning_rate} | "
          f"max_depth={max_depth} | subsample={subsample}")
    _print_metrics(metrics)

    return {
        "model"          : model,
        "y_pred"         : y_pred,
        "y_prob"         : y_prob,
        "metrics"        : metrics,
        "importances"    : importances,
        "actual_n_trees" : actual_trees,
        "train_scores"   : model.train_score_,
    }


# =============================================================================
# 🔧 2. TRAIN GRADIENT BOOSTING REGRESSOR
# =============================================================================

def train_gradient_boosting_regressor(X_train: pd.DataFrame,
                                        X_test: pd.DataFrame,
                                        y_train: pd.Series,
                                        y_test: pd.Series,
                                        n_estimators: int = 200,
                                        learning_rate: float = 0.1,
                                        max_depth: int = 3,
                                        subsample: float = 0.8,
                                        loss: str = "squared_error",
                                        random_state: int = 42) -> dict:
    """
    Trains a Gradient Boosting Regressor.

    Loss functions:
        'squared_error' : L2 loss — sensitive to outliers
        'absolute_error': L1 loss — robust to outliers
        'huber'         : Blend of L1 and L2
        'quantile'      : For quantile regression

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        n_estimators  : Number of boosting trees
        learning_rate : Shrinkage factor (η)
        max_depth     : Max depth per weak learner tree
        subsample     : Fraction of rows per tree
        loss          : Regression loss function
        random_state  : Reproducibility seed

    Returns:
        Dictionary with model, predictions, and regression metrics
    """
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        loss=loss,
        random_state=random_state,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)
    metrics      = _evaluate_regression(y_train, y_pred_train,
                                         y_test, y_pred_test)

    print(f"[GBRegressor] n_estimators={n_estimators} | "
          f"lr={learning_rate} | loss={loss}")
    print(f"  [TRAIN] RMSE={metrics['train']['RMSE']:.4f} | "
          f"R²={metrics['train']['R²']:.4f}")
    print(f"  [TEST ] RMSE={metrics['test']['RMSE']:.4f} | "
          f"R²={metrics['test']['R²']:.4f}")

    return {
        "model"        : model,
        "y_pred_train" : y_pred_train,
        "y_pred_test"  : y_pred_test,
        "metrics"      : metrics,
        "importances"  : pd.Series(model.feature_importances_,
                                    index=X_train.columns)
                          .sort_values(ascending=False),
        "train_scores" : model.train_score_,
    }


# =============================================================================
# 🔧 3. HIST GRADIENT BOOSTING (Faster for Large Datasets)
# =============================================================================

def train_hist_gradient_boosting(X_train: pd.DataFrame,
                                    X_test: pd.DataFrame,
                                    y_train: pd.Series,
                                    y_test: pd.Series,
                                    max_iter: int = 200,
                                    learning_rate: float = 0.1,
                                    max_depth: int = None,
                                    max_leaf_nodes: int = 31,
                                    l2_regularization: float = 0.0,
                                    random_state: int = 42) -> dict:
    """
    Trains HistGradientBoostingClassifier — a faster, histogram-based
    implementation suitable for large datasets (similar to LightGBM).

    Key advantages over GradientBoostingClassifier:
        - Much faster on large datasets (n > 10K)
        - Natively handles missing values
        - Built-in early stopping
        - Supports categorical features

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        max_iter          : Max number of trees (iterations)
        learning_rate     : Shrinkage factor
        max_depth         : Max tree depth (None = unlimited)
        max_leaf_nodes    : Max leaf nodes per tree (controls complexity)
        l2_regularization : L2 penalty on leaf values
        random_state      : Reproducibility seed

    Returns:
        Dictionary with model, predictions, and metrics
    """
    model = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        l2_regularization=l2_regularization,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    model.fit(X_train, y_train)

    y_pred    = model.predict(X_test)
    y_prob    = model.predict_proba(X_test)
    n_classes = len(np.unique(y_train))
    metrics   = _evaluate(y_test, y_pred,
                           y_prob[:, 1] if n_classes == 2 else None,
                           n_classes)

    print(f"[HistGBM] max_iter={max_iter} (actual={model.n_iter_}) | "
          f"lr={learning_rate} | max_leaf_nodes={max_leaf_nodes}")
    _print_metrics(metrics)

    return {
        "model"       : model,
        "y_pred"      : y_pred,
        "y_prob"      : y_prob,
        "metrics"     : metrics,
        "actual_iters": model.n_iter_,
    }


# =============================================================================
# 🔧 4. LEARNING RATE SENSITIVITY
# =============================================================================

def learning_rate_sensitivity(X_train: pd.DataFrame,
                                 X_test: pd.DataFrame,
                                 y_train: pd.Series,
                                 y_test: pd.Series,
                                 lr_values: list = None,
                                 n_estimators: int = 200) -> pd.DataFrame:
    """
    Evaluates GBM performance across a range of learning rates,
    keeping n_estimators fixed — shows the shrinkage vs speed tradeoff.

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        lr_values    : Learning rate values to evaluate
        n_estimators : Fixed number of trees for comparison

    Returns:
        DataFrame with metrics per learning rate
    """
    if lr_values is None:
        lr_values = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    n_classes = len(np.unique(y_train))
    rows = []
    for lr in lr_values:
        model = GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=lr,
            max_depth=3, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_prob  = model.predict_proba(X_test)
        tr_acc  = accuracy_score(y_train, model.predict(X_train))
        te_acc  = accuracy_score(y_test, y_pred)
        te_f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        row = {
            "learning_rate": lr,
            "Train Acc"    : round(tr_acc, 4),
            "Test Acc"     : round(te_acc, 4),
            "Test F1"      : round(te_f1, 4),
            "Gap"          : round(tr_acc - te_acc, 4),
        }
        if n_classes == 2:
            row["ROC-AUC"] = round(roc_auc_score(y_test, y_prob[:, 1]), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Learning Rate Sensitivity (n_estimators={n_estimators}):")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 5. N_ESTIMATORS SENSITIVITY (Training Loss Curve)
# =============================================================================

def n_estimators_sensitivity(X_train: pd.DataFrame,
                                X_test: pd.DataFrame,
                                y_train: pd.Series,
                                y_test: pd.Series,
                                learning_rate: float = 0.1,
                                max_depth: int = 3,
                                n_range: list = None) -> pd.DataFrame:
    """
    Evaluates performance across a range of n_estimators —
    finds the optimal number of boosting rounds.

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        learning_rate : Fixed learning rate
        max_depth     : Fixed max depth
        n_range       : List of n_estimators values to evaluate

    Returns:
        DataFrame with train/test metrics per n_estimators
    """
    if n_range is None:
        n_range = [10, 25, 50, 100, 150, 200, 300, 500]

    n_classes = len(np.unique(y_train))
    rows = []

    for n in n_range:
        model = GradientBoostingClassifier(
            n_estimators=n, learning_rate=learning_rate,
            max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        tr_acc = accuracy_score(y_train, model.predict(X_train))
        te_acc = accuracy_score(y_test, y_pred)

        row = {
            "n_estimators": n,
            "Train Acc"   : round(tr_acc, 4),
            "Test Acc"    : round(te_acc, 4),
            "Gap"         : round(tr_acc - te_acc, 4),
        }
        if n_classes == 2:
            row["ROC-AUC"] = round(roc_auc_score(y_test, y_prob[:, 1]), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    best_n = df.loc[df["Test Acc"].idxmax(), "n_estimators"]
    print(f"n_estimators Sensitivity (lr={learning_rate}, depth={max_depth}):")
    print(df.to_string(index=False))
    print(f"\nOptimal n_estimators: {best_n}")
    return df


# =============================================================================
# 🔧 6. SUBSAMPLE SENSITIVITY (Stochastic GBM)
# =============================================================================

def subsample_sensitivity(X_train: pd.DataFrame,
                            X_test: pd.DataFrame,
                            y_train: pd.Series,
                            y_test: pd.Series,
                            subsample_values: list = None,
                            n_estimators: int = 200,
                            learning_rate: float = 0.1) -> pd.DataFrame:
    """
    Evaluates the effect of row subsampling (stochastic gradient boosting).

    subsample < 1.0 → Each tree uses a random subset of rows
    → Reduces variance → Often improves generalization

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        subsample_values : Subsample fractions to evaluate
        n_estimators     : Fixed n_estimators
        learning_rate    : Fixed learning rate

    Returns:
        DataFrame with metrics per subsample value
    """
    if subsample_values is None:
        subsample_values = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    n_classes = len(np.unique(y_train))
    rows = []

    for ss in subsample_values:
        model = GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=3, subsample=ss, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        te_f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        row = {
            "subsample" : ss,
            "Train Acc" : round(accuracy_score(y_train, model.predict(X_train)), 4),
            "Test Acc"  : round(accuracy_score(y_test, y_pred), 4),
            "Test F1"   : round(te_f1, 4),
        }
        if n_classes == 2:
            row["ROC-AUC"] = round(roc_auc_score(y_test, y_prob[:, 1]), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Subsample Sensitivity (n={n_estimators}, lr={learning_rate}):")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 7. GRIDSEARCHCV / RANDOMIZEDSEARCHCV — TUNING
# =============================================================================

def tune_gradient_boosting(X_train: pd.DataFrame,
                              y_train: pd.Series,
                              method: str = "random",
                              n_iter: int = 30,
                              cv: int = 5,
                              scoring: str = "roc_auc") -> dict:
    """
    Tunes Gradient Boosting hyperparameters using GridSearchCV or
    RandomizedSearchCV.

    Args:
        X_train  : Training features DataFrame
        y_train  : Training target Series
        method   : 'grid' or 'random' (default: 'random')
        n_iter   : Iterations for RandomizedSearchCV
        cv       : Number of CV folds
        scoring  : Scoring metric

    Returns:
        Dictionary with best params, score, and search object
    """
    param_grid = {
        "n_estimators"    : [100, 200, 300, 500],
        "learning_rate"   : [0.01, 0.05, 0.1, 0.2],
        "max_depth"       : [3, 4, 5],
        "subsample"       : [0.7, 0.8, 0.9, 1.0],
        "max_features"    : ["sqrt", "log2", None],
        "min_samples_leaf": [1, 2, 5],
    }

    base_model = GradientBoostingClassifier(random_state=42)

    if method == "grid":
        search = GridSearchCV(
            base_model, param_grid,
            cv=cv, scoring=scoring, n_jobs=-1
        )
    else:
        search = RandomizedSearchCV(
            base_model, param_grid, n_iter=n_iter,
            cv=cv, scoring=scoring, random_state=42, n_jobs=-1
        )

    search.fit(X_train, y_train)

    print(f"[{'GridSearchCV' if method=='grid' else 'RandomizedSearchCV'} GBM] "
          f"Best params: {search.best_params_}")
    print(f"  Best CV {scoring}: {search.best_score_:.4f}")

    return {
        "search"      : search,
        "best_params" : search.best_params_,
        "best_score"  : search.best_score_,
        "best_model"  : search.best_estimator_,
    }


# =============================================================================
# 🔧 8. PERMUTATION IMPORTANCE
# =============================================================================

def compute_permutation_importance(model,
                                     X_test: pd.DataFrame,
                                     y_test: pd.Series,
                                     n_repeats: int = 10,
                                     scoring: str = "accuracy") -> pd.DataFrame:
    """
    Computes Permutation Importance — more reliable than MDI for GBM.

    Args:
        model      : Fitted GradientBoostingClassifier
        X_test     : Test features DataFrame
        y_test     : Test target Series
        n_repeats  : Number of shuffles per feature
        scoring    : Scoring metric

    Returns:
        DataFrame with permutation importance per feature
    """
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats, scoring=scoring,
        random_state=42, n_jobs=-1
    )

    imp_df = pd.DataFrame({
        "Feature"        : X_test.columns,
        "Importance Mean": result.importances_mean,
        "Importance Std" : result.importances_std,
    }).sort_values("Importance Mean", ascending=False).reset_index(drop=True)

    print(f"[Permutation Importance] n_repeats={n_repeats}")
    print(imp_df.round(4).to_string(index=False))
    return imp_df


# =============================================================================
# 🔧 9. TRAINING LOSS CURVE (Deviance / Log-Loss vs Iterations)
# =============================================================================

def get_loss_curve(model: GradientBoostingClassifier) -> pd.DataFrame:
    """
    Extracts the training loss curve from a fitted GBM model.

    The loss curve shows how the training loss (deviance/log-loss)
    decreases with each additional boosting round.

    Args:
        model : Fitted GradientBoostingClassifier

    Returns:
        DataFrame with loss per iteration
    """
    df = pd.DataFrame({
        "Iteration"    : range(1, len(model.train_score_) + 1),
        "Training Loss": model.train_score_,
    })
    print(f"[Loss Curve] {len(df)} iterations")
    print(f"  Initial loss : {df['Training Loss'].iloc[0]:.4f}")
    print(f"  Final loss   : {df['Training Loss'].iloc[-1]:.4f}")
    print(f"  Total drop   : {df['Training Loss'].iloc[0] - df['Training Loss'].iloc[-1]:.4f}")
    return df


# =============================================================================
# 🔧 10. CROSS-VALIDATION
# =============================================================================

def cross_validate_gbm(X: pd.DataFrame,
                         y: pd.Series,
                         n_estimators: int = 200,
                         learning_rate: float = 0.1,
                         max_depth: int = 3,
                         cv: int = 5,
                         scoring: str = "roc_auc") -> dict:
    """
    Performs Stratified K-Fold Cross-Validation on GBM.

    Args:
        X            : Full feature DataFrame
        y            : Full target Series
        n_estimators : Number of trees
        learning_rate: Shrinkage factor
        max_depth    : Max tree depth
        cv           : Number of folds
        scoring      : Scoring metric

    Returns:
        Dictionary with fold scores, mean, and std
    """
    model  = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42
    )
    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)

    result = {
        "scores"      : scores,
        "mean"        : scores.mean(),
        "std"         : scores.std(),
        "fold_results": {f"Fold {i+1}": round(s, 4) for i, s in enumerate(scores)},
    }

    print(f"[GBM CV] n_estimators={n_estimators} | lr={learning_rate} | "
          f"cv={cv} | {scoring}: {scores.mean():.4f} ± {scores.std():.4f}")
    for fold, score in result["fold_results"].items():
        print(f"  {fold}: {score}")

    return result


# =============================================================================
# 🔧 11. EVALUATION
# =============================================================================

def evaluate_gbm_classifier(y_test: pd.Series,
                               y_pred: np.ndarray,
                               y_prob: np.ndarray = None,
                               model_name: str = "Gradient Boosting") -> pd.DataFrame:
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
        metrics["Log Loss"] = round(log_loss(y_test, y_prob), 4)

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

    # ── 1. Gradient Boosting Classifier ────────────────────────────────────
    print("\n" + "=" * 65)
    print("1️⃣  Gradient Boosting Classifier")
    print("=" * 65)
    result = train_gradient_boosting(X_train, X_test, y_train, y_test)
    evaluate_gbm_classifier(y_test, result["y_pred"], result["y_prob"][:, 1])

    # ── 2. Loss Curve ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("2️⃣  Training Loss Curve")
    print("=" * 65)
    loss_df = get_loss_curve(result["model"])

    # ── 3. Learning Rate Sensitivity ───────────────────────────────────────
    print("\n" + "=" * 65)
    print("3️⃣  Learning Rate Sensitivity")
    print("=" * 65)
    lr_df = learning_rate_sensitivity(X_train, X_test, y_train, y_test)

    # ── 4. n_estimators Sensitivity ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("4️⃣  n_estimators Sensitivity")
    print("=" * 65)
    n_df = n_estimators_sensitivity(X_train, X_test, y_train, y_test)

    # ── 5. Subsample Sensitivity ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("5️⃣  Subsample Sensitivity (Stochastic GBM)")
    print("=" * 65)
    ss_df = subsample_sensitivity(X_train, X_test, y_train, y_test)

    # ── 6. Permutation Importance ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("6️⃣  Permutation Importance")
    print("=" * 65)
    perm_df = compute_permutation_importance(result["model"], X_test, y_test)

    # ── 7. Hist Gradient Boosting (fast) ───────────────────────────────────
    print("\n" + "=" * 65)
    print("7️⃣  HistGradientBoosting (Fast Version)")
    print("=" * 65)
    hist_result = train_hist_gradient_boosting(X_train, X_test, y_train, y_test)

    # ── 8. RandomizedSearchCV ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("8️⃣  RandomizedSearchCV — Hyperparameter Tuning")
    print("=" * 65)
    search_result = tune_gradient_boosting(X_train, y_train,
                                            method="random", n_iter=20)

    # ── 9. Cross-Validation ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("9️⃣  Stratified 5-Fold CV")
    print("=" * 65)
    cv_result = cross_validate_gbm(X, y, n_estimators=200, learning_rate=0.1)

    print("\n✅ All Gradient Boosting techniques demonstrated successfully!")
