# =============================================================================
# 📦 XGBoost — Reusable ML Script
# =============================================================================
# Author   : Hamna Munir
# Topic    : 03_Supervised_Learning / Classification / XGBoost
# File     : xgboost.py
# =============================================================================

# -----------------------------------------------------------------------------
# 📚 Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from xgboost import XGBClassifier, XGBRegressor
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
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 🔧 1. TRAIN XGBOOST CLASSIFIER (sklearn API)
# =============================================================================

def train_xgboost_classifier(X_train: pd.DataFrame,
                               X_test: pd.DataFrame,
                               y_train: pd.Series,
                               y_test: pd.Series,
                               n_estimators: int = 300,
                               learning_rate: float = 0.1,
                               max_depth: int = 5,
                               subsample: float = 0.8,
                               colsample_bytree: float = 0.8,
                               reg_alpha: float = 0.0,
                               reg_lambda: float = 1.0,
                               gamma: float = 0.0,
                               min_child_weight: int = 1,
                               scale_pos_weight: float = 1.0,
                               early_stopping_rounds: int = 50,
                               eval_metric: str = "logloss",
                               random_state: int = 42) -> dict:
    """
    Trains an XGBoost Classifier using the sklearn-compatible API.

    Key advantages over sklearn GBM:
        - Second-order gradient optimization (Newton's method)
        - Built-in L1 (alpha) + L2 (lambda) + gamma regularization
        - Missing value handling natively
        - Column subsampling at tree / level / node level
        - Early stopping to prevent overfitting

    Note: XGBoost does NOT require feature scaling — tree-based splits
    are scale-invariant.

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        n_estimators       : Max number of boosting rounds
        learning_rate      : Shrinkage factor eta (typical: 0.01–0.3)
        max_depth          : Max tree depth (typical: 3–7)
        subsample          : Row subsampling fraction (0.5–1.0)
        colsample_bytree   : Feature subsampling per tree (0.5–1.0)
        reg_alpha          : L1 regularization on leaf weights
        reg_lambda         : L2 regularization on leaf weights (default: 1.0)
        gamma              : Min loss reduction to split (complexity penalty)
        min_child_weight   : Min sum of hessian in a child node
        scale_pos_weight   : Imbalance correction = n_negative / n_positive
        early_stopping_rounds : Stop if val metric doesn't improve
        eval_metric        : Validation metric for early stopping
        random_state       : Reproducibility seed

    Returns:
        Dictionary with model, predictions, probabilities, and metrics
    """
    n_classes = len(np.unique(y_train))
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"

    # Validation set for early stopping (20% of train)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15,
        stratify=y_train, random_state=random_state
    )

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        gamma=gamma,
        min_child_weight=min_child_weight,
        scale_pos_weight=scale_pos_weight,
        objective=objective,
        eval_metric=eval_metric,
        early_stopping_rounds=early_stopping_rounds,
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    y_pred    = model.predict(X_test)
    y_prob    = model.predict_proba(X_test)
    metrics   = _evaluate(y_test, y_pred,
                           y_prob[:, 1] if n_classes == 2 else None,
                           n_classes)

    importances = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    best_iter = model.best_iteration
    print(f"[XGBClassifier] n_estimators={n_estimators} (best={best_iter}) | "
          f"lr={learning_rate} | max_depth={max_depth}")
    print(f"  reg_alpha={reg_alpha} | reg_lambda={reg_lambda} | "
          f"gamma={gamma} | subsample={subsample}")
    _print_metrics(metrics)

    return {
        "model"       : model,
        "y_pred"      : y_pred,
        "y_prob"      : y_prob,
        "metrics"     : metrics,
        "importances" : importances,
        "best_iter"   : best_iter,
    }


# =============================================================================
# 🔧 2. TRAIN XGBOOST REGRESSOR
# =============================================================================

def train_xgboost_regressor(X_train: pd.DataFrame,
                              X_test: pd.DataFrame,
                              y_train: pd.Series,
                              y_test: pd.Series,
                              n_estimators: int = 300,
                              learning_rate: float = 0.1,
                              max_depth: int = 5,
                              subsample: float = 0.8,
                              colsample_bytree: float = 0.8,
                              reg_alpha: float = 0.0,
                              reg_lambda: float = 1.0,
                              early_stopping_rounds: int = 50,
                              random_state: int = 42) -> dict:
    """
    Trains an XGBoost Regressor.

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        n_estimators       : Max boosting rounds
        learning_rate      : Shrinkage factor
        max_depth          : Max tree depth
        subsample          : Row subsampling
        colsample_bytree   : Feature subsampling per tree
        reg_alpha          : L1 regularization
        reg_lambda         : L2 regularization
        early_stopping_rounds : Early stop patience
        random_state       : Reproducibility seed

    Returns:
        Dictionary with model, predictions, and regression metrics
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=random_state
    )

    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=early_stopping_rounds,
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(X_tr, y_tr,
               eval_set=[(X_val, y_val)],
               verbose=False)

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)
    metrics      = _evaluate_regression(y_train, y_pred_train,
                                         y_test,  y_pred_test)

    print(f"[XGBRegressor] best_iter={model.best_iteration} | "
          f"lr={learning_rate} | max_depth={max_depth}")
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
        "best_iter"    : model.best_iteration,
    }


# =============================================================================
# 🔧 3. REGULARIZATION SENSITIVITY (alpha, lambda, gamma)
# =============================================================================

def regularization_sensitivity(X_train: pd.DataFrame,
                                  X_test: pd.DataFrame,
                                  y_train: pd.Series,
                                  y_test: pd.Series,
                                  param: str = "reg_lambda",
                                  values: list = None) -> pd.DataFrame:
    """
    Evaluates XGBoost performance across a range of regularization values.

    Params:
        reg_alpha  : L1 — promotes sparsity in leaf weights
        reg_lambda : L2 — smoothly shrinks leaf weights (default: 1.0)
        gamma      : Min gain to split — penalizes tree complexity

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        param  : Regularization parameter to sweep ('reg_alpha',
                 'reg_lambda', or 'gamma')
        values : List of values to evaluate

    Returns:
        DataFrame with metrics per parameter value
    """
    if values is None:
        values = [0, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]

    n_classes = len(np.unique(y_train))
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15,
        stratify=y_train, random_state=42
    )

    rows = []
    for val in values:
        kwargs = {
            "n_estimators": 200, "learning_rate": 0.1,
            "max_depth": 5, "subsample": 0.8,
            "colsample_bytree": 0.8, "early_stopping_rounds": 30,
            "tree_method": "hist", "random_state": 42,
            "n_jobs": -1, "verbosity": 0,
        }
        kwargs[param] = val

        model = XGBClassifier(**kwargs)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        y_pred  = model.predict(X_test)
        y_prob  = model.predict_proba(X_test)
        tr_acc  = accuracy_score(y_train, model.predict(X_train))
        te_acc  = accuracy_score(y_test, y_pred)
        te_f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        row = {
            param     : val,
            "Train Acc": round(tr_acc, 4),
            "Test Acc" : round(te_acc, 4),
            "Test F1"  : round(te_f1, 4),
            "Best Iter": model.best_iteration,
        }
        if n_classes == 2:
            row["ROC-AUC"] = round(roc_auc_score(y_test, y_prob[:, 1]), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Regularization Sensitivity ({param}):")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 4. LEARNING RATE + N_ESTIMATORS SENSITIVITY
# =============================================================================

def lr_estimators_sensitivity(X_train: pd.DataFrame,
                                 X_test: pd.DataFrame,
                                 y_train: pd.Series,
                                 y_test: pd.Series,
                                 configs: list = None) -> pd.DataFrame:
    """
    Evaluates XGBoost across different (learning_rate, n_estimators) pairs.

    Low lr + many trees typically outperforms high lr + few trees,
    but takes longer to train.

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        configs : List of (learning_rate, n_estimators) tuples

    Returns:
        DataFrame with metrics per configuration
    """
    if configs is None:
        configs = [
            (0.3,  100),
            (0.1,  200),
            (0.05, 400),
            (0.01, 1000),
        ]

    n_classes = len(np.unique(y_train))
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15,
        stratify=y_train, random_state=42
    )

    rows = []
    for lr, n_est in configs:
        model = XGBClassifier(
            n_estimators=n_est, learning_rate=lr,
            max_depth=5, subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=30, tree_method="hist",
            random_state=42, n_jobs=-1, verbosity=0
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        te_acc = accuracy_score(y_test, y_pred)
        te_f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        row = {
            "learning_rate": lr,
            "n_estimators" : n_est,
            "Best Iter"    : model.best_iteration,
            "Test Acc"     : round(te_acc, 4),
            "Test F1"      : round(te_f1, 4),
        }
        if n_classes == 2:
            row["ROC-AUC"] = round(roc_auc_score(y_test, y_prob[:, 1]), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    print("Learning Rate × n_estimators Trade-off:")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 5. FEATURE IMPORTANCE — THREE TYPES
# =============================================================================

def get_feature_importance(model: XGBClassifier,
                              feature_names: list,
                              importance_type: str = "weight") -> pd.DataFrame:
    """
    Extracts XGBoost feature importance by one of three methods.

    XGBoost provides three built-in importance types:
        'weight'  : Number of times a feature is used in splits (frequency)
        'gain'    : Average gain of splits using the feature (most informative)
        'cover'   : Average coverage of splits using the feature (sample count)

    Args:
        model          : Fitted XGBClassifier
        feature_names  : List of feature column names
        importance_type: 'weight', 'gain', or 'cover'

    Returns:
        DataFrame with feature importances
    """
    scores = model.get_booster().get_score(importance_type=importance_type)

    imp_df = pd.DataFrame({
        "Feature"   : list(scores.keys()),
        "Importance": list(scores.values()),
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    imp_df["Importance %"] = (
        imp_df["Importance"] / imp_df["Importance"].sum() * 100
    ).round(2)

    print(f"[Feature Importance — {importance_type}]")
    print(imp_df.head(15).round(4).to_string(index=False))
    return imp_df


# =============================================================================
# 🔧 6. COMPARE IMPORTANCE TYPES
# =============================================================================

def compare_importance_types(model: XGBClassifier,
                               X_test: pd.DataFrame,
                               y_test: pd.Series) -> pd.DataFrame:
    """
    Compares feature rankings by weight, gain, cover, and
    permutation importance.

    Args:
        model  : Fitted XGBClassifier
        X_test : Test features DataFrame
        y_test : Test target Series

    Returns:
        DataFrame with rank per feature per importance type
    """
    types = ["weight", "gain", "cover"]
    rank_df = pd.DataFrame(index=X_test.columns)

    for imp_type in types:
        scores = model.get_booster().get_score(importance_type=imp_type)
        series = pd.Series(scores)
        rank_df[f"{imp_type}_rank"] = series.rank(ascending=False)

    # Permutation importance
    perm = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    perm_series = pd.Series(perm.importances_mean, index=X_test.columns)
    rank_df["perm_rank"] = perm_series.rank(ascending=False)

    rank_df = rank_df.dropna().sort_values("gain_rank")
    print("Feature Importance Comparison (ranks, lower = more important):")
    print(rank_df.round(1).to_string())
    return rank_df


# =============================================================================
# 🔧 7. GRIDSEARCHCV / RANDOMIZEDSEARCHCV — TUNING
# =============================================================================

def tune_xgboost(X_train: pd.DataFrame,
                  y_train: pd.Series,
                  method: str = "random",
                  n_iter: int = 30,
                  cv: int = 5,
                  scoring: str = "roc_auc") -> dict:
    """
    Tunes XGBoost hyperparameters using GridSearchCV or RandomizedSearchCV.

    Args:
        X_train : Training features DataFrame
        y_train : Training target Series
        method  : 'grid' or 'random' (default: 'random')
        n_iter  : Iterations for RandomizedSearchCV
        cv      : Number of CV folds
        scoring : Scoring metric

    Returns:
        Dictionary with best params, score, and search object
    """
    param_grid = {
        "n_estimators"     : [100, 200, 300, 500],
        "learning_rate"    : [0.01, 0.05, 0.1, 0.2],
        "max_depth"        : [3, 4, 5, 6, 7],
        "subsample"        : [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree" : [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha"        : [0, 0.01, 0.1, 1.0],
        "reg_lambda"       : [0.5, 1.0, 2.0, 5.0],
        "gamma"            : [0, 0.1, 0.5, 1.0],
        "min_child_weight" : [1, 3, 5, 10],
    }

    base_model = XGBClassifier(
        tree_method="hist", random_state=42,
        n_jobs=-1, verbosity=0
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

    print(f"[{'GridSearchCV' if method=='grid' else 'RandomizedSearchCV'} XGB] "
          f"Best params: {search.best_params_}")
    print(f"  Best CV {scoring}: {search.best_score_:.4f}")

    return {
        "search"      : search,
        "best_params" : search.best_params_,
        "best_score"  : search.best_score_,
        "best_model"  : search.best_estimator_,
    }


# =============================================================================
# 🔧 8. HANDLE CLASS IMBALANCE (scale_pos_weight)
# =============================================================================

def train_xgb_imbalanced(X_train: pd.DataFrame,
                           X_test: pd.DataFrame,
                           y_train: pd.Series,
                           y_test: pd.Series,
                           n_estimators: int = 300,
                           learning_rate: float = 0.1,
                           max_depth: int = 5) -> dict:
    """
    Trains XGBoost with automatic class imbalance correction.

    scale_pos_weight = n_negative / n_positive
    → Upweights the minority (positive) class during training
    → Improves recall for the minority class

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        n_estimators : Number of trees
        learning_rate: Shrinkage factor
        max_depth    : Max tree depth

    Returns:
        Dictionary comparing default vs scale_pos_weight models
    """
    n_neg  = (y_train == 0).sum()
    n_pos  = (y_train == 1).sum()
    spw    = n_neg / n_pos

    print(f"Class balance: 0={n_neg} | 1={n_pos} | "
          f"scale_pos_weight={spw:.2f}")

    results = {}

    for name, spw_val in [("Default (spw=1)", 1.0),
                           (f"Balanced (spw={spw:.2f})", spw)]:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.15,
            stratify=y_train, random_state=42
        )
        model = XGBClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw_val, early_stopping_rounds=30,
            tree_method="hist", random_state=42,
            n_jobs=-1, verbosity=0
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = model.predict(X_test)
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
# 🔧 9. CROSS-VALIDATION
# =============================================================================

def cross_validate_xgb(X: pd.DataFrame,
                         y: pd.Series,
                         n_estimators: int = 200,
                         learning_rate: float = 0.1,
                         max_depth: int = 5,
                         cv: int = 5,
                         scoring: str = "roc_auc") -> dict:
    """
    Performs Stratified K-Fold Cross-Validation on XGBoost.

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
    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)

    result = {
        "scores"      : scores,
        "mean"        : scores.mean(),
        "std"         : scores.std(),
        "fold_results": {f"Fold {i+1}": round(s, 4) for i, s in enumerate(scores)},
    }

    print(f"[XGB CV] n_estimators={n_estimators} | lr={learning_rate} | "
          f"cv={cv} | {scoring}: {scores.mean():.4f} ± {scores.std():.4f}")
    for fold, score in result["fold_results"].items():
        print(f"  {fold}: {score}")

    return result


# =============================================================================
# 🔧 10. EVALUATION
# =============================================================================

def evaluate_xgb_classifier(y_test: pd.Series,
                               y_pred: np.ndarray,
                               y_prob: np.ndarray = None,
                               model_name: str = "XGBoost") -> pd.DataFrame:
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

    # ── Synthetic classification dataset ──────────────────────────────────
    X_raw, y_raw = make_classification(
        n_samples=1200, n_features=15, n_informative=8,
        n_redundant=3, n_classes=2, weights=[0.75, 0.25],
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

    # ── 1. XGBoost Classifier ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("1️⃣  XGBoost Classifier")
    print("=" * 65)
    result = train_xgboost_classifier(X_train, X_test, y_train, y_test)
    evaluate_xgb_classifier(y_test, result["y_pred"], result["y_prob"][:, 1])

    # ── 2. Feature Importance (three types) ───────────────────────────────
    print("\n" + "=" * 65)
    print("2️⃣  Feature Importance — Gain")
    print("=" * 65)
    imp_df = get_feature_importance(result["model"],
                                     X_train.columns.tolist(),
                                     importance_type="gain")

    # ── 3. Compare importance types ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("3️⃣  Compare Importance Types (weight / gain / cover / perm)")
    print("=" * 65)
    rank_df = compare_importance_types(result["model"], X_test, y_test)

    # ── 4. Regularization Sensitivity ─────────────────────────────────────
    print("\n" + "=" * 65)
    print("4️⃣  Regularization Sensitivity (reg_lambda)")
    print("=" * 65)
    reg_df = regularization_sensitivity(X_train, X_test, y_train, y_test,
                                         param="reg_lambda")

    # ── 5. LR × n_estimators trade-off ────────────────────────────────────
    print("\n" + "=" * 65)
    print("5️⃣  Learning Rate × n_estimators Trade-off")
    print("=" * 65)
    lr_df = lr_estimators_sensitivity(X_train, X_test, y_train, y_test)

    # ── 6. Class Imbalance (scale_pos_weight) ─────────────────────────────
    print("\n" + "=" * 65)
    print("6️⃣  Class Imbalance — scale_pos_weight")
    print("=" * 65)
    imb_result = train_xgb_imbalanced(X_train, X_test, y_train, y_test)

    # ── 7. RandomizedSearchCV ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("7️⃣  RandomizedSearchCV — Tuning")
    print("=" * 65)
    search_result = tune_xgboost(X_train, y_train, method="random", n_iter=20)

    # ── 8. Cross-Validation ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("8️⃣  Stratified 5-Fold CV")
    print("=" * 65)
    cv_result = cross_validate_xgb(X, y)

    # ── 9. XGBoost Regressor ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("9️⃣  XGBoost Regressor")
    print("=" * 65)
    from sklearn.datasets import make_regression
    X_reg_raw, y_reg_raw = make_regression(
        n_samples=800, n_features=10, noise=20, random_state=42
    )
    X_reg = pd.DataFrame(X_reg_raw, columns=[f"F{i+1}" for i in range(10)])
    y_reg = pd.Series(y_reg_raw, name="Target")
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    reg_result = train_xgboost_regressor(Xr_tr, Xr_te, yr_tr, yr_te)

    print("\n✅ All XGBoost techniques demonstrated successfully!")
