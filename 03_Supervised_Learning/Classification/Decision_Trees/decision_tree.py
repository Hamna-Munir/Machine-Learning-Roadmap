# =============================================================================
# 📦 Decision Tree — Reusable ML Script
# =============================================================================
# Author   : Hamna Munir
# Topic    : 03_Supervised_Learning / Classification / Decision_Trees
# File     : decision_tree.py
# =============================================================================

# -----------------------------------------------------------------------------
# 📚 Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
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

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 🔧 1. TRAIN DECISION TREE CLASSIFIER
# =============================================================================

def train_decision_tree(X_train: pd.DataFrame,
                          X_test: pd.DataFrame,
                          y_train: pd.Series,
                          y_test: pd.Series,
                          max_depth: int = 5,
                          min_samples_split: int = 2,
                          min_samples_leaf: int = 1,
                          criterion: str = "gini",
                          class_weight=None,
                          random_state: int = 42) -> dict:
    """
    Trains a Decision Tree Classifier.

    Note: Decision Trees do NOT require feature scaling — they split
    on raw feature values regardless of scale.

    Args:
        X_train           : Training features DataFrame
        X_test            : Test features DataFrame
        y_train            : Training target Series
        y_test             : Test target Series
        max_depth          : Maximum tree depth (default: 5, prevents overfitting)
        min_samples_split  : Min samples to split a node (default: 2)
        min_samples_leaf   : Min samples required in a leaf (default: 1)
        criterion          : Splitting criterion — 'gini' or 'entropy'
        class_weight       : 'balanced' for imbalanced classes, or None
        random_state       : Reproducibility seed

    Returns:
        Dictionary with model, predictions, probabilities, and metrics
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        class_weight=class_weight,
        random_state=random_state
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

    print(f"[DecisionTree] max_depth={max_depth} | criterion={criterion} | "
          f"min_samples_leaf={min_samples_leaf}")
    print(f"  Tree depth (actual) : {model.get_depth()}")
    print(f"  Number of leaves    : {model.get_n_leaves()}")
    _print_metrics(metrics)

    return {
        "model"      : model,
        "y_pred"     : y_pred,
        "y_prob"     : y_prob,
        "metrics"    : metrics,
        "importances": importances,
        "depth"      : model.get_depth(),
        "n_leaves"   : model.get_n_leaves(),
    }


# =============================================================================
# 🔧 2. TRAIN DECISION TREE REGRESSOR
# =============================================================================

def train_decision_tree_regressor(X_train: pd.DataFrame,
                                    X_test: pd.DataFrame,
                                    y_train: pd.Series,
                                    y_test: pd.Series,
                                    max_depth: int = 5,
                                    min_samples_split: int = 2,
                                    min_samples_leaf: int = 1,
                                    criterion: str = "squared_error",
                                    random_state: int = 42) -> dict:
    """
    Trains a Decision Tree Regressor.

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        max_depth          : Maximum tree depth
        min_samples_split  : Min samples to split a node
        min_samples_leaf   : Min samples required in a leaf
        criterion          : 'squared_error', 'absolute_error', 'friedman_mse'
        random_state       : Reproducibility seed

    Returns:
        Dictionary with model, predictions, and regression metrics
    """
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)
    metrics      = _evaluate_regression(y_train, y_pred_train, y_test, y_pred_test)

    print(f"[DecisionTreeRegressor] max_depth={max_depth} | criterion={criterion}")
    print(f"  [TRAIN] RMSE={metrics['train']['RMSE']:.4f} | R²={metrics['train']['R²']:.4f}")
    print(f"  [TEST ] RMSE={metrics['test']['RMSE']:.4f} | R²={metrics['test']['R²']:.4f}")

    return {
        "model"       : model,
        "y_pred_train": y_pred_train,
        "y_pred_test" : y_pred_test,
        "metrics"     : metrics,
        "importances" : pd.Series(model.feature_importances_, index=X_train.columns)
                          .sort_values(ascending=False),
    }


# =============================================================================
# 🔧 3. MAX_DEPTH SENSITIVITY ANALYSIS
# =============================================================================

def depth_sensitivity(X_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        y_train: pd.Series,
                        y_test: pd.Series,
                        depth_range: list = None) -> pd.DataFrame:
    """
    Evaluates train/test performance across a range of max_depth values —
    reveals underfitting (shallow trees) and overfitting (deep trees).

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        depth_range : List of max_depth values to evaluate

    Returns:
        DataFrame with train/test accuracy per depth
    """
    if depth_range is None:
        depth_range = list(range(1, 21))

    rows = []
    for depth in depth_range:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        tr_acc = accuracy_score(y_train, model.predict(X_train))
        te_acc = accuracy_score(y_test,  model.predict(X_test))
        rows.append({
            "max_depth": depth,
            "Train Acc": round(tr_acc, 4),
            "Test Acc" : round(te_acc, 4),
            "Gap"      : round(tr_acc - te_acc, 4),
            "N Leaves" : model.get_n_leaves(),
        })

    df = pd.DataFrame(rows)
    best_depth = df.loc[df["Test Acc"].idxmax(), "max_depth"]
    print(f"Depth Sensitivity Analysis:")
    print(df.to_string(index=False))
    print(f"\nOptimal max_depth (best test accuracy): {best_depth}")
    return df


# =============================================================================
# 🔧 4. COST COMPLEXITY PRUNING PATH
# =============================================================================

def cost_complexity_pruning(X_train: pd.DataFrame,
                              X_test: pd.DataFrame,
                              y_train: pd.Series,
                              y_test: pd.Series,
                              random_state: int = 42) -> dict:
    """
    Computes the cost-complexity pruning path and finds the optimal
    ccp_alpha value that maximizes test accuracy.

    Post-pruning approach: grow full tree, then prune back using alpha.
    Larger alpha = more aggressive pruning = simpler tree.

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        random_state : Reproducibility seed

    Returns:
        Dictionary with pruning path, accuracies, and best alpha
    """
    full_tree = DecisionTreeClassifier(random_state=random_state)
    path      = full_tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    impurities = path.impurities

    # Remove the largest alpha (trivial tree with 1 node)
    ccp_alphas = ccp_alphas[:-1] if len(ccp_alphas) > 1 else ccp_alphas

    train_scores, test_scores, n_nodes_list = [], [], []
    for alpha in ccp_alphas:
        model = DecisionTreeClassifier(ccp_alpha=alpha, random_state=random_state)
        model.fit(X_train, y_train)
        train_scores.append(accuracy_score(y_train, model.predict(X_train)))
        test_scores.append(accuracy_score(y_test, model.predict(X_test)))
        n_nodes_list.append(model.tree_.node_count)

    best_idx   = np.argmax(test_scores)
    best_alpha = ccp_alphas[best_idx]

    print(f"[Cost Complexity Pruning] {len(ccp_alphas)} candidate alphas")
    print(f"  Best ccp_alpha : {best_alpha:.6f}")
    print(f"  Test accuracy at best alpha: {test_scores[best_idx]:.4f}")
    print(f"  Tree nodes at best alpha   : {n_nodes_list[best_idx]}")

    return {
        "ccp_alphas"   : ccp_alphas,
        "train_scores" : train_scores,
        "test_scores"  : test_scores,
        "n_nodes"      : n_nodes_list,
        "best_alpha"   : best_alpha,
        "best_test_acc": test_scores[best_idx],
    }


# =============================================================================
# 🔧 5. GRIDSEARCHCV — DECISION TREE TUNING
# =============================================================================

def tune_decision_tree(X_train: pd.DataFrame,
                         y_train: pd.Series,
                         task: str = "classification",
                         cv: int = 5,
                         scoring: str = "f1_weighted") -> dict:
    """
    Tunes Decision Tree hyperparameters using GridSearchCV.

    Searches over:
        - max_depth
        - min_samples_split
        - min_samples_leaf
        - criterion

    Args:
        X_train : Training features DataFrame
        y_train : Training target Series
        task    : 'classification' or 'regression'
        cv      : Number of CV folds
        scoring : Scoring metric

    Returns:
        Dictionary with best params, score, and GridSearchCV object
    """
    if task == "classification":
        model = DecisionTreeClassifier(random_state=42)
        param_grid = {
            "max_depth"        : [3, 5, 7, 10, 15, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf" : [1, 2, 5, 10],
            "criterion"        : ["gini", "entropy"],
        }
    else:
        model = DecisionTreeRegressor(random_state=42)
        param_grid = {
            "max_depth"        : [3, 5, 7, 10, 15, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf" : [1, 2, 5, 10],
        }
        if scoring == "f1_weighted":
            scoring = "r2"

    grid = GridSearchCV(
        model, param_grid,
        cv=cv, scoring=scoring, n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print(f"[GridSearchCV DecisionTree] Best params: {grid.best_params_}")
    print(f"  Best CV {scoring}: {grid.best_score_:.4f}")

    return {
        "grid"       : grid,
        "best_params": grid.best_params_,
        "best_score" : grid.best_score_,
        "best_model" : grid.best_estimator_,
    }


# =============================================================================
# 🔧 6. CRITERION COMPARISON (Gini vs Entropy)
# =============================================================================

def compare_criteria(X_train: pd.DataFrame,
                       X_test: pd.DataFrame,
                       y_train: pd.Series,
                       y_test: pd.Series,
                       max_depth: int = 5) -> pd.DataFrame:
    """
    Compares Gini Impurity vs Entropy (Information Gain) splitting criteria.

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        max_depth : Fixed depth for fair comparison

    Returns:
        DataFrame with performance per criterion
    """
    rows = []
    for criterion in ["gini", "entropy"]:
        model = DecisionTreeClassifier(
            max_depth=max_depth, criterion=criterion, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        n_classes = len(np.unique(y_train))

        row = {
            "Criterion": criterion,
            "Accuracy" : round(accuracy_score(y_test, y_pred), 4),
            "F1"       : round(f1_score(y_test, y_pred,
                                        average="weighted", zero_division=0), 4),
            "N Leaves" : model.get_n_leaves(),
            "Depth"    : model.get_depth(),
        }
        if n_classes == 2:
            row["ROC-AUC"] = round(roc_auc_score(y_test, y_prob[:, 1]), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Criterion Comparison (max_depth={max_depth}):")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 7. FEATURE IMPORTANCE
# =============================================================================

def get_feature_importance(model: DecisionTreeClassifier,
                             feature_names: list,
                             top_n: int = 10) -> pd.DataFrame:
    """
    Extracts and ranks feature importances from a fitted Decision Tree.

    Importance = total weighted impurity decrease attributed to each feature
    across all splits where it was used.

    Args:
        model         : Fitted DecisionTreeClassifier/Regressor
        feature_names : List of feature column names
        top_n         : Number of top features to display

    Returns:
        DataFrame with features ranked by importance
    """
    importances = pd.DataFrame({
        "Feature"   : feature_names,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    importances["Cumulative %"] = (
        importances["Importance"].cumsum() / importances["Importance"].sum() * 100
    ).round(2)

    print(f"\n[Feature Importance] Top {top_n} features:")
    print(importances.head(top_n).round(4).to_string(index=False))
    return importances


# =============================================================================
# 🔧 8. EXPORT TREE AS TEXT
# =============================================================================

def export_tree_rules(model: DecisionTreeClassifier,
                        feature_names: list) -> str:
    """
    Exports the decision tree as human-readable text rules.

    Args:
        model         : Fitted DecisionTreeClassifier
        feature_names : List of feature column names

    Returns:
        String representation of the tree structure
    """
    tree_text = export_text(model, feature_names=list(feature_names))
    print("[Decision Tree Rules]")
    print(tree_text)
    return tree_text


# =============================================================================
# 🔧 9. CROSS-VALIDATION
# =============================================================================

def cross_validate_tree(X: pd.DataFrame,
                          y: pd.Series,
                          max_depth: int = 5,
                          cv: int = 5,
                          scoring: str = "f1_weighted") -> dict:
    """
    Performs Stratified K-Fold Cross-Validation on a Decision Tree.

    Args:
        X         : Full feature DataFrame
        y         : Full target Series
        max_depth : Tree depth (default: 5)
        cv        : Number of folds
        scoring   : Scoring metric

    Returns:
        Dictionary with fold scores, mean, and std
    """
    model  = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)

    result = {
        "scores"      : scores,
        "mean"        : scores.mean(),
        "std"         : scores.std(),
        "fold_results": {f"Fold {i+1}": round(s, 4) for i, s in enumerate(scores)},
    }

    print(f"[DecisionTree CV] max_depth={max_depth} | cv={cv} | "
          f"{scoring}: {scores.mean():.4f} ± {scores.std():.4f}")
    for fold, score in result["fold_results"].items():
        print(f"  {fold}: {score}")

    return result


# =============================================================================
# 🔧 10. EVALUATION METRICS
# =============================================================================

def evaluate_tree_classifier(y_test: pd.Series,
                                y_pred: np.ndarray,
                                y_prob: np.ndarray = None,
                                model_name: str = "Decision Tree") -> pd.DataFrame:
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
        n_samples=800, n_features=10, n_informative=6,
        n_redundant=2, n_classes=2, weights=[0.7, 0.3],
        random_state=42
    )
    X = pd.DataFrame(X_raw, columns=[f"Feature_{i+1}" for i in range(10)])
    y = pd.Series(y_raw, name="Target")

    print("=" * 65)
    print("📊 Dataset Info")
    print("=" * 65)
    print(f"Shape: {X.shape} | Classes: {dict(y.value_counts().sort_index())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── 1. Train basic tree ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("1️⃣  Decision Tree (max_depth=5, gini)")
    print("=" * 65)
    result = train_decision_tree(X_train, X_test, y_train, y_test, max_depth=5)

    # ── 2. Feature importance ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("2️⃣  Feature Importance")
    print("=" * 65)
    importance_df = get_feature_importance(result["model"], X_train.columns.tolist())

    # ── 3. Depth sensitivity ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("3️⃣  Max Depth Sensitivity")
    print("=" * 65)
    depth_df = depth_sensitivity(X_train, X_test, y_train, y_test)

    # ── 4. Cost complexity pruning ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("4️⃣  Cost Complexity Pruning")
    print("=" * 65)
    pruning_result = cost_complexity_pruning(X_train, X_test, y_train, y_test)

    # ── 5. Criterion comparison ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("5️⃣  Gini vs Entropy")
    print("=" * 65)
    criteria_df = compare_criteria(X_train, X_test, y_train, y_test)

    # ── 6. GridSearchCV ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("6️⃣  GridSearchCV — Full Tuning")
    print("=" * 65)
    gs_result = tune_decision_tree(X_train, y_train)

    # ── 7. Cross-Validation ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("7️⃣  Stratified 5-Fold CV")
    print("=" * 65)
    cv_result = cross_validate_tree(X, y, max_depth=5)

    # ── 8. Export tree rules ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("8️⃣  Export Tree as Text Rules")
    print("=" * 65)
    small_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    small_tree.fit(X_train, y_train)
    rules = export_tree_rules(small_tree, X_train.columns.tolist())

    print("\n✅ All Decision Tree techniques demonstrated successfully!")
