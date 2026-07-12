# =============================================================================
# 📦 CatBoost — Reusable ML Script
# =============================================================================
# Author   : Hamna Munir
# Topic    : 03_Supervised_Learning / Classification / CatBoost
# File     : catboost.py
# =============================================================================

# -----------------------------------------------------------------------------
# 📚 Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

try:
    from catboost import (
        CatBoostClassifier, CatBoostRegressor,
        Pool, cv as catboost_cv
    )
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️  CatBoost not installed. Run: pip install catboost")

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

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 🔧 1. TRAIN CATBOOST CLASSIFIER
# =============================================================================

def train_catboost_classifier(X_train: pd.DataFrame,
                                X_test: pd.DataFrame,
                                y_train: pd.Series,
                                y_test: pd.Series,
                                cat_features: list = None,
                                iterations: int = 500,
                                learning_rate: float = 0.05,
                                depth: int = 6,
                                l2_leaf_reg: float = 3.0,
                                rsm: float = 1.0,
                                subsample: float = 1.0,
                                early_stopping_rounds: int = 50,
                                eval_metric: str = "AUC",
                                class_weights=None,
                                random_seed: int = 42) -> dict:
    """
    Trains a CatBoost Classifier.

    Key advantages:
        - Pass categorical columns directly — no encoding needed
        - Ordered Target Statistics for leak-free categorical encoding
        - Ordered Boosting — prevents prediction shift / overfitting
        - Symmetric trees — fast prediction, good regularization
        - Minimal hyperparameter tuning needed

    Note: CatBoost does NOT require feature scaling.
          Pass categorical feature names/indices via cat_features.

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        cat_features       : List of column names or indices that are categorical
                             These are passed RAW (strings/ints) — no encoding!
        iterations         : Max number of trees (use early stopping)
        learning_rate      : Shrinkage factor (default: auto if not set)
        depth              : Tree depth (symmetric trees, typical: 6–10)
        l2_leaf_reg        : L2 regularization on leaf values (default: 3.0)
        rsm                : Feature subsampling fraction (like colsample_bytree)
        subsample          : Row subsampling fraction
        early_stopping_rounds : Stop if eval_metric doesn't improve
        eval_metric        : Validation metric — 'AUC', 'F1', 'Logloss', 'Accuracy'
        class_weights      : Dict {class: weight} for imbalanced data
        random_seed        : Reproducibility seed

    Returns:
        Dictionary with model, predictions, probabilities, and metrics
    """
    if not CATBOOST_AVAILABLE:
        raise ImportError("CatBoost is not installed. Run: pip install catboost")

    if cat_features is None:
        # Auto-detect categorical columns from DataFrame
        cat_features = X_train.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        if cat_features:
            print(f"  Auto-detected categorical features: {cat_features}")

    # Validation set for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15,
        stratify=y_train, random_state=random_seed
    )

    n_classes = len(np.unique(y_train))
    loss_func = "Logloss" if n_classes == 2 else "MultiClass"

    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        rsm=rsm,
        subsample=subsample,
        loss_function=loss_func,
        eval_metric=eval_metric,
        early_stopping_rounds=early_stopping_rounds,
        class_weights=class_weights,
        random_seed=random_seed,
        verbose=0,
        thread_count=-1,
    )

    model.fit(
        X_tr, y_tr,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        verbose=False,
    )

    y_pred    = model.predict(X_test).ravel()
    y_prob    = model.predict_proba(X_test)
    metrics   = _evaluate(y_test, y_pred,
                           y_prob[:, 1] if n_classes == 2 else None,
                           n_classes)

    importances = pd.Series(
        model.get_feature_importance(),
        index=X_train.columns
    ).sort_values(ascending=False)

    best_iter = model.get_best_iteration()
    print(f"[CatBoostClassifier] iterations={iterations} (best={best_iter}) | "
          f"lr={learning_rate} | depth={depth}")
    print(f"  l2_leaf_reg={l2_leaf_reg} | cat_features={cat_features}")
    _print_metrics(metrics)

    return {
        "model"      : model,
        "y_pred"     : y_pred,
        "y_prob"     : y_prob,
        "metrics"    : metrics,
        "importances": importances,
        "best_iter"  : best_iter,
        "cat_features": cat_features,
    }


# =============================================================================
# 🔧 2. TRAIN CATBOOST REGRESSOR
# =============================================================================

def train_catboost_regressor(X_train: pd.DataFrame,
                               X_test: pd.DataFrame,
                               y_train: pd.Series,
                               y_test: pd.Series,
                               cat_features: list = None,
                               iterations: int = 500,
                               learning_rate: float = 0.05,
                               depth: int = 6,
                               l2_leaf_reg: float = 3.0,
                               loss_function: str = "RMSE",
                               early_stopping_rounds: int = 50,
                               random_seed: int = 42) -> dict:
    """
    Trains a CatBoost Regressor.

    Loss functions:
        'RMSE'     : Root Mean Squared Error (default)
        'MAE'      : Mean Absolute Error (robust to outliers)
        'Huber'    : Blend of RMSE and MAE
        'Quantile' : Quantile regression

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        cat_features       : List of categorical feature names/indices
        iterations         : Max boosting rounds
        learning_rate      : Shrinkage factor
        depth              : Symmetric tree depth
        l2_leaf_reg        : L2 regularization on leaf values
        loss_function      : Regression loss function
        early_stopping_rounds : Early stop patience
        random_seed        : Reproducibility seed

    Returns:
        Dictionary with model, predictions, and regression metrics
    """
    if not CATBOOST_AVAILABLE:
        raise ImportError("CatBoost is not installed. Run: pip install catboost")

    if cat_features is None:
        cat_features = X_train.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=random_seed
    )

    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        loss_function=loss_function,
        eval_metric=loss_function,
        early_stopping_rounds=early_stopping_rounds,
        random_seed=random_seed,
        verbose=0,
        thread_count=-1,
    )

    model.fit(
        X_tr, y_tr,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        verbose=False,
    )

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)
    metrics      = _evaluate_regression(y_train, y_pred_train,
                                         y_test, y_pred_test)

    print(f"[CatBoostRegressor] best_iter={model.get_best_iteration()} | "
          f"lr={learning_rate} | depth={depth}")
    print(f"  [TRAIN] RMSE={metrics['train']['RMSE']:.4f} | "
          f"R²={metrics['train']['R²']:.4f}")
    print(f"  [TEST ] RMSE={metrics['test']['RMSE']:.4f} | "
          f"R²={metrics['test']['R²']:.4f}")

    return {
        "model"        : model,
        "y_pred_train" : y_pred_train,
        "y_pred_test"  : y_pred_test,
        "metrics"      : metrics,
        "importances"  : pd.Series(model.get_feature_importance(),
                                    index=X_train.columns)
                          .sort_values(ascending=False),
        "best_iter"    : model.get_best_iteration(),
    }


# =============================================================================
# 🔧 3. FEATURE IMPORTANCE — MULTIPLE TYPES
# =============================================================================

def get_feature_importance(model: "CatBoostClassifier",
                              X_test: pd.DataFrame,
                              y_test: pd.Series = None,
                              cat_features: list = None,
                              importance_type: str = "PredictionValuesChange"
                              ) -> pd.DataFrame:
    """
    Extracts CatBoost feature importance by one of four methods.

    CatBoost importance types:
        'PredictionValuesChange' : How much predictions change (default, fast)
        'LossFunctionChange'     : How much loss changes if feature removed
                                   (more accurate, slower)
        'ShapValues'             : SHAP additive attribution (most trustworthy)
        'Interaction'            : Pairwise feature interaction strength

    Args:
        model          : Fitted CatBoostClassifier or Regressor
        X_test         : Test features DataFrame (needed for some types)
        y_test         : Test labels (needed for LossFunctionChange)
        cat_features   : List of categorical feature names
        importance_type: One of the four types above

    Returns:
        DataFrame with feature importances
    """
    pool = Pool(X_test, label=y_test, cat_features=cat_features or [])

    if importance_type == "ShapValues":
        shap_vals = model.get_feature_importance(pool, type="ShapValues")
        # Last column is bias — exclude it
        importances = np.abs(shap_vals[:, :-1]).mean(axis=0)
    else:
        importances = model.get_feature_importance(pool, type=importance_type)

    imp_df = pd.DataFrame({
        "Feature"    : X_test.columns,
        "Importance" : importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    imp_df["Importance %"] = (
        imp_df["Importance"] / imp_df["Importance"].sum() * 100
    ).round(2)
    imp_df["Cumulative %"] = imp_df["Importance %"].cumsum().round(2)

    print(f"[Feature Importance — {importance_type}] Top 15:")
    print(imp_df.head(15).round(4).to_string(index=False))
    return imp_df


# =============================================================================
# 🔧 4. DEPTH SENSITIVITY
# =============================================================================

def depth_sensitivity(X_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        y_train: pd.Series,
                        y_test: pd.Series,
                        cat_features: list = None,
                        depth_range: list = None,
                        learning_rate: float = 0.05) -> pd.DataFrame:
    """
    Evaluates CatBoost performance across a range of tree depths.

    CatBoost uses symmetric (oblivious) trees — depth controls the number
    of unique conditions in the tree (2^depth leaf nodes max).
    Typical best range: 6–10 (deeper than LightGBM/XGBoost due to symmetry).

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        cat_features  : Categorical feature names
        depth_range   : List of depth values to evaluate
        learning_rate : Fixed learning rate for comparison

    Returns:
        DataFrame with metrics per depth value
    """
    if depth_range is None:
        depth_range = [3, 4, 5, 6, 7, 8, 10]

    if cat_features is None:
        cat_features = X_train.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    n_classes = len(np.unique(y_train))
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15,
        stratify=y_train, random_state=42
    )

    rows = []
    for d in depth_range:
        model = CatBoostClassifier(
            iterations=300, learning_rate=learning_rate,
            depth=d, l2_leaf_reg=3.0,
            early_stopping_rounds=30,
            random_seed=42, verbose=0, thread_count=-1
        )
        model.fit(X_tr, y_tr, cat_features=cat_features,
                  eval_set=(X_val, y_val), verbose=False)

        y_pred  = model.predict(X_test).ravel()
        y_prob  = model.predict_proba(X_test)
        tr_acc  = accuracy_score(y_train, model.predict(X_train).ravel())
        te_acc  = accuracy_score(y_test, y_pred)
        te_f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        row = {
            "depth"    : d,
            "Train Acc": round(tr_acc, 4),
            "Test Acc" : round(te_acc, 4),
            "Test F1"  : round(te_f1, 4),
            "Gap"      : round(tr_acc - te_acc, 4),
            "Best Iter": model.get_best_iteration(),
        }
        if n_classes == 2:
            row["ROC-AUC"] = round(roc_auc_score(y_test, y_prob[:, 1]), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    best_d = df.loc[df["Test Acc"].idxmax(), "depth"]
    print(f"Depth Sensitivity (lr={learning_rate}):")
    print(df.to_string(index=False))
    print(f"\nOptimal depth: {best_d}")
    return df


# =============================================================================
# 🔧 5. L2 REGULARIZATION SENSITIVITY
# =============================================================================

def l2_sensitivity(X_train: pd.DataFrame,
                    X_test: pd.DataFrame,
                    y_train: pd.Series,
                    y_test: pd.Series,
                    cat_features: list = None,
                    l2_values: list = None) -> pd.DataFrame:
    """
    Evaluates CatBoost performance across a range of l2_leaf_reg values.

    l2_leaf_reg controls L2 regularization on leaf values —
    prevents overfitting by penalizing large leaf weights.

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        cat_features : Categorical feature names
        l2_values    : List of l2_leaf_reg values to evaluate

    Returns:
        DataFrame with metrics per l2_leaf_reg value
    """
    if l2_values is None:
        l2_values = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]

    if cat_features is None:
        cat_features = X_train.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    n_classes = len(np.unique(y_train))
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15,
        stratify=y_train, random_state=42
    )

    rows = []
    for l2 in l2_values:
        model = CatBoostClassifier(
            iterations=300, learning_rate=0.05,
            depth=6, l2_leaf_reg=l2,
            early_stopping_rounds=30,
            random_seed=42, verbose=0, thread_count=-1
        )
        model.fit(X_tr, y_tr, cat_features=cat_features,
                  eval_set=(X_val, y_val), verbose=False)

        y_pred = model.predict(X_test).ravel()
        y_prob = model.predict_proba(X_test)
        te_acc = accuracy_score(y_test, y_pred)
        te_f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        row = {
            "l2_leaf_reg": l2,
            "Test Acc"   : round(te_acc, 4),
            "Test F1"    : round(te_f1, 4),
            "Best Iter"  : model.get_best_iteration(),
        }
        if n_classes == 2:
            row["ROC-AUC"] = round(roc_auc_score(y_test, y_prob[:, 1]), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    print("l2_leaf_reg Sensitivity:")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 6. CATBOOST CV (Native)
# =============================================================================

def catboost_native_cv(X_train: pd.DataFrame,
                         y_train: pd.Series,
                         cat_features: list = None,
                         params: dict = None,
                         fold_count: int = 5,
                         num_boost_round: int = 300,
                         early_stopping_rounds: int = 50) -> pd.DataFrame:
    """
    Runs CatBoost's native cross-validation — more efficient than
    sklearn CV for CatBoost because it uses the Pool data format
    and supports early stopping per fold.

    Args:
        X_train           : Training features DataFrame
        y_train           : Training target Series
        cat_features      : Categorical feature names
        params            : CatBoost params dict (auto-set if None)
        fold_count        : Number of CV folds
        num_boost_round   : Max boosting rounds
        early_stopping_rounds : Early stopping patience

    Returns:
        DataFrame with CV results per iteration
    """
    if cat_features is None:
        cat_features = X_train.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    if params is None:
        n_classes = len(np.unique(y_train))
        params = {
            "iterations"    : num_boost_round,
            "learning_rate" : 0.05,
            "depth"         : 6,
            "l2_leaf_reg"   : 3.0,
            "loss_function" : "Logloss" if n_classes == 2 else "MultiClass",
            "eval_metric"   : "AUC"     if n_classes == 2 else "Accuracy",
            "random_seed"   : 42,
            "verbose"       : 0,
            "thread_count"  : -1,
        }

    pool = Pool(X_train, label=y_train, cat_features=cat_features)

    cv_results = catboost_cv(
        pool=pool,
        params=params,
        fold_count=fold_count,
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
        plot=False,
    )

    best_iter = cv_results.index[
        cv_results.iloc[:, 2].idxmax()  # test metric column
    ]

    print(f"[CatBoost Native CV] fold_count={fold_count}")
    print(f"  Columns: {cv_results.columns.tolist()}")
    print(f"  Best iteration: {best_iter}")
    print(f"  Final row:\n{cv_results.tail(1).to_string()}")
    return cv_results


# =============================================================================
# 🔧 7. GRIDSEARCHCV — CATBOOST TUNING
# =============================================================================

def tune_catboost(X_train: pd.DataFrame,
                   y_train: pd.Series,
                   cat_features: list = None,
                   method: str = "random",
                   n_iter: int = 20,
                   cv: int = 5,
                   scoring: str = "roc_auc") -> dict:
    """
    Tunes CatBoost hyperparameters using GridSearchCV or RandomizedSearchCV.

    Note: For CatBoost, native CV + manual grid search is often better,
    but sklearn API works when cat_features has been label-encoded.

    Args:
        X_train     : Training features (encode cats first for sklearn CV)
        y_train     : Training target Series
        cat_features: Categorical feature names (for encoding note)
        method      : 'grid' or 'random' (default: 'random')
        n_iter      : Iterations for RandomizedSearchCV
        cv          : Number of CV folds
        scoring     : Scoring metric

    Returns:
        Dictionary with best params, score, and search object
    """
    param_grid = {
        "iterations"   : [200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "depth"        : [4, 5, 6, 7, 8],
        "l2_leaf_reg"  : [1.0, 3.0, 5.0, 10.0],
        "rsm"          : [0.7, 0.8, 0.9, 1.0],
        "subsample"    : [0.7, 0.8, 0.9, 1.0],
    }

    base_model = CatBoostClassifier(
        random_seed=42, verbose=0, thread_count=-1
    )

    if method == "grid":
        search = GridSearchCV(
            base_model, param_grid,
            cv=cv, scoring=scoring, n_jobs=-1
        )
    else:
        search = RandomizedSearchCV(
            base_model, param_grid, n_iter=n_iter,
            cv=cv, scoring=scoring,
            random_state=42, n_jobs=-1
        )

    search.fit(X_train, y_train)

    print(f"[{'GridSearchCV' if method=='grid' else 'RandomizedSearchCV'} "
          f"CatBoost] Best params: {search.best_params_}")
    print(f"  Best CV {scoring}: {search.best_score_:.4f}")

    return {
        "search"      : search,
        "best_params" : search.best_params_,
        "best_score"  : search.best_score_,
        "best_model"  : search.best_estimator_,
    }


# =============================================================================
# 🔧 8. CLASS IMBALANCE — auto_class_weights
# =============================================================================

def train_catboost_imbalanced(X_train: pd.DataFrame,
                                X_test: pd.DataFrame,
                                y_train: pd.Series,
                                y_test: pd.Series,
                                cat_features: list = None,
                                iterations: int = 300,
                                learning_rate: float = 0.05) -> dict:
    """
    Compares CatBoost strategies for handling class imbalance.

    CatBoost options:
        Default                : No rebalancing
        auto_class_weights='Balanced' : Auto-compute class weights
        class_weights={0: w0, 1: w1}  : Manual class weights

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        cat_features : Categorical feature names
        iterations   : Number of trees
        learning_rate: Shrinkage factor

    Returns:
        Dictionary comparing all three strategies
    """
    if cat_features is None:
        cat_features = X_train.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw   = n_neg / n_pos

    print(f"Class balance: 0={n_neg} | 1={n_pos} | "
          f"imbalance_ratio={spw:.2f}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15,
        stratify=y_train, random_state=42
    )

    configs = [
        ("Default",                       {}),
        ("auto_class_weights='Balanced'", {"auto_class_weights": "Balanced"}),
        (f"class_weights={{0:1, 1:{spw:.1f}}}",
         {"class_weights": {0: 1.0, 1: float(spw)}}),
    ]

    results = {}
    for name, extra_params in configs:
        model = CatBoostClassifier(
            iterations=iterations, learning_rate=learning_rate,
            depth=6, l2_leaf_reg=3.0, early_stopping_rounds=30,
            random_seed=42, verbose=0, thread_count=-1,
            **extra_params
        )
        model.fit(X_tr, y_tr, cat_features=cat_features,
                  eval_set=(X_val, y_val), verbose=False)

        y_pred = model.predict(X_test).ravel()
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Accuracy" : round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall"   : round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1"       : round(f1_score(y_test, y_pred, zero_division=0), 4),
            "ROC-AUC"  : round(roc_auc_score(y_test, y_prob), 4),
        }
        results[name] = {"model": model, "metrics": metrics}
        print(f"\n  {name}: {metrics}")

    comp = pd.DataFrame({k: v["metrics"] for k, v in results.items()}).T
    print("\nComparison:")
    print(comp.to_string())
    return results


# =============================================================================
# 🔧 9. CROSS-VALIDATION (sklearn)
# =============================================================================

def cross_validate_catboost(X: pd.DataFrame,
                              y: pd.Series,
                              cat_features: list = None,
                              iterations: int = 300,
                              learning_rate: float = 0.05,
                              depth: int = 6,
                              cv: int = 5,
                              scoring: str = "roc_auc") -> dict:
    """
    Performs Stratified K-Fold Cross-Validation on CatBoost.

    Args:
        X            : Full feature DataFrame
        y            : Full target Series
        cat_features : Categorical feature names
        iterations   : Number of trees
        learning_rate: Shrinkage factor
        depth        : Tree depth
        cv           : Number of folds
        scoring      : Scoring metric

    Returns:
        Dictionary with fold scores, mean, and std
    """
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
        thread_count=-1,
    )
    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)

    result = {
        "scores"      : scores,
        "mean"        : scores.mean(),
        "std"         : scores.std(),
        "fold_results": {f"Fold {i+1}": round(s, 4) for i, s in enumerate(scores)},
    }

    print(f"[CatBoost CV] iterations={iterations} | lr={learning_rate} | "
          f"depth={depth} | cv={cv}")
    print(f"  {scoring}: {scores.mean():.4f} ± {scores.std():.4f}")
    for fold, score in result["fold_results"].items():
        print(f"  {fold}: {score}")

    return result


# =============================================================================
# 🔧 10. EVALUATION
# =============================================================================

def evaluate_catboost_classifier(y_test: pd.Series,
                                   y_pred: np.ndarray,
                                   y_prob: np.ndarray = None,
                                   model_name: str = "CatBoost") -> pd.DataFrame:
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
# 🚀 MAIN — Demo with Synthetic Dataset (including categorical features)
# =============================================================================

if __name__ == "__main__":

    if not CATBOOST_AVAILABLE:
        print("Install CatBoost: pip install catboost")
        exit()

    from sklearn.datasets import make_classification

    np.random.seed(42)

    # ── Synthetic dataset with categorical features ────────────────────────
    X_num_raw, y_raw = make_classification(
        n_samples=1000, n_features=10, n_informative=6,
        n_redundant=2, n_classes=2, weights=[0.70, 0.30],
        random_state=42
    )

    X = pd.DataFrame(X_num_raw,
                     columns=[f"Num_{i+1}" for i in range(10)])

    # Add synthetic categorical features
    X["City"]      = np.random.choice(["London","Paris","Berlin","Tokyo","Sydney"], 1000)
    X["Education"] = np.random.choice(["HighSchool","Bachelor","Master","PhD"], 1000,
                                        p=[0.25,0.40,0.25,0.10])
    X["Gender"]    = np.random.choice(["Male","Female"], 1000)

    y = pd.Series(y_raw, name="Target")
    cat_cols = ["City", "Education", "Gender"]

    print("=" * 65)
    print("📊 Dataset Info — With Categorical Features")
    print("=" * 65)
    print(f"Shape    : {X.shape}")
    print(f"Cat cols : {cat_cols}")
    print(f"Classes  : {dict(y.value_counts().sort_index())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── 1. CatBoost Classifier ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("1️⃣  CatBoost Classifier (with native categorical features)")
    print("=" * 65)
    result = train_catboost_classifier(
        X_train, X_test, y_train, y_test, cat_features=cat_cols
    )
    evaluate_catboost_classifier(
        y_test, result["y_pred"], result["y_prob"][:, 1]
    )

    # ── 2. Feature Importance ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("2️⃣  Feature Importance — PredictionValuesChange")
    print("=" * 65)
    imp_df = get_feature_importance(
        result["model"], X_test, y_test, cat_features=cat_cols
    )

    # ── 3. Depth Sensitivity ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("3️⃣  Depth Sensitivity")
    print("=" * 65)
    depth_df = depth_sensitivity(
        X_train, X_test, y_train, y_test, cat_features=cat_cols
    )

    # ── 4. L2 Regularization Sensitivity ─────────────────────────────────
    print("\n" + "=" * 65)
    print("4️⃣  l2_leaf_reg Sensitivity")
    print("=" * 65)
    l2_df = l2_sensitivity(
        X_train, X_test, y_train, y_test, cat_features=cat_cols
    )

    # ── 5. Class Imbalance ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("5️⃣  Class Imbalance — auto_class_weights")
    print("=" * 65)
    imb_result = train_catboost_imbalanced(
        X_train, X_test, y_train, y_test, cat_features=cat_cols
    )

    # ── 6. RandomizedSearchCV ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("6️⃣  RandomizedSearchCV — Tuning (numerical features only)")
    print("=" * 65)
    # Note: sklearn CV works best with numerical-encoded cats
    X_num_only = X[[c for c in X.columns if c not in cat_cols]]
    X_tr_num, _, y_tr_num, _ = train_test_split(
        X_num_only, y, test_size=0.2, stratify=y, random_state=42
    )
    search_result = tune_catboost(X_tr_num, y_tr_num,
                                   method="random", n_iter=15)

    # ── 7. Cross-Validation ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("7️⃣  Stratified 5-Fold CV (numerical features only)")
    print("=" * 65)
    cv_result = cross_validate_catboost(X_num_only, y)

    # ── 8. CatBoost Regressor ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("8️⃣  CatBoost Regressor")
    print("=" * 65)
    from sklearn.datasets import make_regression
    X_reg_raw, y_reg_raw = make_regression(
        n_samples=800, n_features=8, noise=20, random_state=42
    )
    X_reg = pd.DataFrame(X_reg_raw, columns=[f"F{i+1}" for i in range(8)])
    X_reg["Category"] = np.random.choice(["A","B","C","D"], 800)
    y_reg = pd.Series(y_reg_raw, name="Target")

    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    reg_result = train_catboost_regressor(
        Xr_tr, Xr_te, yr_tr, yr_te, cat_features=["Category"]
    )

    print("\n✅ All CatBoost techniques demonstrated successfully!")
