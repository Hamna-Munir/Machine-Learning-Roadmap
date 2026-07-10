# =============================================================================
# 📦 LightGBM — Reusable ML Script
# =============================================================================
# Author   : Hamna Munir
# Topic    : 03_Supervised_Learning / Classification / LightGBM
# File     : lightgbm.py
# =============================================================================

# -----------------------------------------------------------------------------
# 📚 Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor

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
# 🔧 1. TRAIN LIGHTGBM CLASSIFIER (sklearn API)
# =============================================================================

def train_lgbm_classifier(X_train: pd.DataFrame,
                            X_test: pd.DataFrame,
                            y_train: pd.Series,
                            y_test: pd.Series,
                            n_estimators: int = 500,
                            learning_rate: float = 0.05,
                            num_leaves: int = 63,
                            max_depth: int = -1,
                            subsample: float = 0.8,
                            colsample_bytree: float = 0.8,
                            min_child_samples: int = 20,
                            reg_alpha: float = 0.0,
                            reg_lambda: float = 0.0,
                            class_weight=None,
                            early_stopping_rounds: int = 50,
                            random_state: int = 42) -> dict:
    """
    Trains a LightGBM Classifier using the sklearn-compatible API.

    Key innovations vs XGBoost:
        - Histogram-based splits → much faster on large datasets
        - Leaf-wise tree growth (best-first) → lower loss per leaf
        - GOSS: Gradient-based One-Side Sampling
        - EFB: Exclusive Feature Bundling
        - Native categorical feature support

    Note: LightGBM does NOT require feature scaling.

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        n_estimators       : Max boosting rounds (use early stopping)
        learning_rate      : Shrinkage factor (typical: 0.01–0.1)
        num_leaves         : Most important param — controls complexity
                             (typical: 20–300, default: 31)
        max_depth          : Max tree depth (-1 = unlimited)
                             Use num_leaves instead of max_depth
        subsample          : Row subsampling per tree (bagging_fraction)
        colsample_bytree   : Feature subsampling per tree (feature_fraction)
        min_child_samples  : Min samples per leaf (min_data_in_leaf)
                             Increase for large datasets to prevent overfit
        reg_alpha          : L1 regularization (lambda_l1)
        reg_lambda         : L2 regularization (lambda_l2)
        class_weight       : 'balanced' for imbalanced classes, or None
        early_stopping_rounds : Stop if val metric doesn't improve
        random_state       : Reproducibility seed

    Returns:
        Dictionary with model, predictions, probabilities, and metrics
    """
    n_classes  = len(np.unique(y_train))
    objective  = "binary" if n_classes == 2 else "multiclass"
    eval_metric = "binary_logloss" if n_classes == 2 else "multi_logloss"

    # Validation set for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15,
        stratify=y_train, random_state=random_state
    )

    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        subsample=subsample,
        subsample_freq=1,
        colsample_bytree=colsample_bytree,
        min_child_samples=min_child_samples,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        class_weight=class_weight,
        objective=objective,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    callbacks = [
        lgb.early_stopping(early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=-1),
    ]

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric=eval_metric,
        callbacks=callbacks,
    )

    y_pred    = model.predict(X_test)
    y_prob    = model.predict_proba(X_test)
    metrics   = _evaluate(y_test, y_pred,
                           y_prob[:, 1] if n_classes == 2 else None,
                           n_classes)

    importances = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    best_iter = model.best_iteration_
    print(f"[LGBMClassifier] n_estimators={n_estimators} (best={best_iter}) | "
          f"lr={learning_rate} | num_leaves={num_leaves}")
    print(f"  subsample={subsample} | colsample_bytree={colsample_bytree} | "
          f"min_child_samples={min_child_samples}")
    _print_metrics(metrics)

    return {
        "model"      : model,
        "y_pred"     : y_pred,
        "y_prob"     : y_prob,
        "metrics"    : metrics,
        "importances": importances,
        "best_iter"  : best_iter,
    }


# =============================================================================
# 🔧 2. TRAIN LIGHTGBM REGRESSOR
# =============================================================================

def train_lgbm_regressor(X_train: pd.DataFrame,
                           X_test: pd.DataFrame,
                           y_train: pd.Series,
                           y_test: pd.Series,
                           n_estimators: int = 500,
                           learning_rate: float = 0.05,
                           num_leaves: int = 63,
                           subsample: float = 0.8,
                           colsample_bytree: float = 0.8,
                           min_child_samples: int = 20,
                           reg_alpha: float = 0.0,
                           reg_lambda: float = 0.0,
                           early_stopping_rounds: int = 50,
                           random_state: int = 42) -> dict:
    """
    Trains a LightGBM Regressor.

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        n_estimators      : Max boosting rounds
        learning_rate     : Shrinkage factor
        num_leaves        : Max leaves per tree
        subsample         : Row subsampling fraction
        colsample_bytree  : Feature subsampling per tree
        min_child_samples : Min samples per leaf
        reg_alpha         : L1 regularization
        reg_lambda        : L2 regularization
        early_stopping_rounds : Early stop patience
        random_state      : Reproducibility seed

    Returns:
        Dictionary with model, predictions, and regression metrics
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=random_state
    )

    model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=subsample,
        subsample_freq=1,
        colsample_bytree=colsample_bytree,
        min_child_samples=min_child_samples,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective="regression",
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    callbacks = [
        lgb.early_stopping(early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=-1),
    ]

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=callbacks,
    )

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)
    metrics      = _evaluate_regression(y_train, y_pred_train,
                                         y_test, y_pred_test)

    print(f"[LGBMRegressor] best_iter={model.best_iteration_} | "
          f"lr={learning_rate} | num_leaves={num_leaves}")
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
        "best_iter"    : model.best_iteration_,
    }


# =============================================================================
# 🔧 3. NATIVE LGBM API (More Control)
# =============================================================================

def train_lgbm_native(X_train: pd.DataFrame,
                        X_test: pd.DataFrame,
                        y_train: pd.Series,
                        y_test: pd.Series,
                        params: dict = None,
                        num_boost_round: int = 1000,
                        early_stopping_rounds: int = 50) -> dict:
    """
    Trains LightGBM using the native lgb.train() API —
    provides more granular control than the sklearn wrapper.

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        params             : LightGBM parameter dict (auto-set if None)
        num_boost_round    : Max boosting rounds
        early_stopping_rounds : Early stop patience

    Returns:
        Dictionary with model, predictions, and metrics
    """
    if params is None:
        n_classes = len(np.unique(y_train))
        params = {
            "objective"   : "binary" if n_classes == 2 else "multiclass",
            "metric"      : "binary_logloss" if n_classes == 2 else "multi_logloss",
            "num_leaves"  : 63,
            "learning_rate": 0.05,
            "subsample"   : 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "reg_alpha"   : 0.0,
            "reg_lambda"  : 0.0,
            "n_jobs"      : -1,
            "verbose"     : -1,
        }
        if n_classes > 2:
            params["num_class"] = n_classes

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15,
        stratify=y_train, random_state=42
    )

    dtrain  = lgb.Dataset(X_tr,  label=y_tr)
    dval    = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    callbacks = [
        lgb.early_stopping(early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=-1),
    ]

    model = lgb.train(
        params, dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dval],
        callbacks=callbacks,
    )

    n_classes = len(np.unique(y_train))
    y_pred_prob = model.predict(X_test)

    if n_classes == 2:
        y_prob = y_pred_prob
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_prob = y_pred_prob
        y_pred = np.argmax(y_prob, axis=1)

    metrics = _evaluate(y_test, y_pred,
                         y_prob if n_classes == 2 else None,
                         n_classes)

    print(f"[LightGBM Native API] best_iteration={model.best_iteration}")
    _print_metrics(metrics)

    return {
        "model"      : model,
        "y_pred"     : y_pred,
        "y_prob"     : y_prob,
        "metrics"    : metrics,
        "best_iter"  : model.best_iteration,
    }


# =============================================================================
# 🔧 4. NUM_LEAVES SENSITIVITY
# =============================================================================

def num_leaves_sensitivity(X_train: pd.DataFrame,
                             X_test: pd.DataFrame,
                             y_train: pd.Series,
                             y_test: pd.Series,
                             leaves_range: list = None,
                             learning_rate: float = 0.05) -> pd.DataFrame:
    """
    Evaluates LightGBM performance across a range of num_leaves values.

    num_leaves is the MOST IMPORTANT hyperparameter in LightGBM.
    Controls model complexity via leaf-wise tree growth.

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        leaves_range  : List of num_leaves values to evaluate
        learning_rate : Fixed learning rate for comparison

    Returns:
        DataFrame with metrics per num_leaves value
    """
    if leaves_range is None:
        leaves_range = [7, 15, 31, 63, 127, 255, 511]

    n_classes = len(np.unique(y_train))
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15,
        stratify=y_train, random_state=42
    )

    rows = []
    for nl in leaves_range:
        model = LGBMClassifier(
            n_estimators=500, learning_rate=learning_rate,
            num_leaves=nl, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, random_state=42,
            n_jobs=-1, verbose=-1
        )
        callbacks = [lgb.early_stopping(30, verbose=False),
                     lgb.log_evaluation(period=-1)]
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=callbacks)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        tr_acc = accuracy_score(y_train, model.predict(X_train))
        te_acc = accuracy_score(y_test, y_pred)

        row = {
            "num_leaves": nl,
            "Train Acc" : round(tr_acc, 4),
            "Test Acc"  : round(te_acc, 4),
            "Gap"       : round(tr_acc - te_acc, 4),
            "Best Iter" : model.best_iteration_,
        }
        if n_classes == 2:
            row["ROC-AUC"] = round(roc_auc_score(y_test, y_prob[:, 1]), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    best_nl = df.loc[df["Test Acc"].idxmax(), "num_leaves"]
    print(f"num_leaves Sensitivity (lr={learning_rate}):")
    print(df.to_string(index=False))
    print(f"\nOptimal num_leaves: {best_nl}")
    return df


# =============================================================================
# 🔧 5. FEATURE IMPORTANCE — THREE TYPES
# =============================================================================

def get_feature_importance(model: LGBMClassifier,
                              feature_names: list,
                              importance_type: str = "gain") -> pd.DataFrame:
    """
    Extracts LightGBM feature importance by one of two methods.

    LightGBM importance types:
        'split' : Number of times a feature is used in a split (frequency)
        'gain'  : Total gain (impurity reduction) from splits using this feature

    Args:
        model          : Fitted LGBMClassifier or LGBMRegressor
        feature_names  : List of feature column names
        importance_type: 'split' or 'gain' (default: 'gain')

    Returns:
        DataFrame with feature importances sorted descending
    """
    importances = model.booster_.feature_importance(importance_type=importance_type)

    imp_df = pd.DataFrame({
        "Feature"    : feature_names,
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
# 🔧 6. PERMUTATION IMPORTANCE
# =============================================================================

def compute_permutation_importance(model: LGBMClassifier,
                                     X_test: pd.DataFrame,
                                     y_test: pd.Series,
                                     n_repeats: int = 10,
                                     scoring: str = "accuracy") -> pd.DataFrame:
    """
    Computes Permutation Importance for a fitted LightGBM model.

    More reliable than built-in split/gain importance, as it is
    evaluated on the test set and not biased by feature cardinality.

    Args:
        model      : Fitted LGBMClassifier
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
# 🔧 7. GRIDSEARCHCV / RANDOMIZEDSEARCHCV — TUNING
# =============================================================================

def tune_lgbm(X_train: pd.DataFrame,
               y_train: pd.Series,
               method: str = "random",
               n_iter: int = 30,
               cv: int = 5,
               scoring: str = "roc_auc") -> dict:
    """
    Tunes LightGBM hyperparameters using GridSearchCV or RandomizedSearchCV.

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
        "n_estimators"     : [200, 300, 500],
        "learning_rate"    : [0.01, 0.05, 0.1],
        "num_leaves"       : [31, 63, 127, 255],
        "subsample"        : [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree" : [0.6, 0.7, 0.8, 0.9],
        "min_child_samples": [10, 20, 50, 100],
        "reg_alpha"        : [0.0, 0.1, 0.5, 1.0],
        "reg_lambda"       : [0.0, 0.1, 0.5, 1.0],
    }

    base_model = LGBMClassifier(
        random_state=42, n_jobs=-1, verbose=-1
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

    print(f"[{'GridSearchCV' if method=='grid' else 'RandomizedSearchCV'} LGBM] "
          f"Best params: {search.best_params_}")
    print(f"  Best CV {scoring}: {search.best_score_:.4f}")

    return {
        "search"      : search,
        "best_params" : search.best_params_,
        "best_score"  : search.best_score_,
        "best_model"  : search.best_estimator_,
    }


# =============================================================================
# 🔧 8. CROSS-VALIDATION
# =============================================================================

def cross_validate_lgbm(X: pd.DataFrame,
                          y: pd.Series,
                          n_estimators: int = 300,
                          learning_rate: float = 0.05,
                          num_leaves: int = 63,
                          cv: int = 5,
                          scoring: str = "roc_auc") -> dict:
    """
    Performs Stratified K-Fold Cross-Validation on LightGBM.

    Args:
        X            : Full feature DataFrame
        y            : Full target Series
        n_estimators : Number of trees
        learning_rate: Shrinkage factor
        num_leaves   : Max leaves per tree
        cv           : Number of folds
        scoring      : Scoring metric

    Returns:
        Dictionary with fold scores, mean, and std
    """
    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)

    result = {
        "scores"      : scores,
        "mean"        : scores.mean(),
        "std"         : scores.std(),
        "fold_results": {f"Fold {i+1}": round(s, 4) for i, s in enumerate(scores)},
    }

    print(f"[LGBM CV] n_estimators={n_estimators} | lr={learning_rate} | "
          f"num_leaves={num_leaves} | cv={cv}")
    print(f"  {scoring}: {scores.mean():.4f} ± {scores.std():.4f}")
    for fold, score in result["fold_results"].items():
        print(f"  {fold}: {score}")

    return result


# =============================================================================
# 🔧 9. CLASS IMBALANCE — is_unbalance / scale_pos_weight
# =============================================================================

def train_lgbm_imbalanced(X_train: pd.DataFrame,
                            X_test: pd.DataFrame,
                            y_train: pd.Series,
                            y_test: pd.Series,
                            n_estimators: int = 300,
                            learning_rate: float = 0.05) -> dict:
    """
    Compares LightGBM strategies for handling class imbalance.

    LightGBM provides two options:
        is_unbalance=True       → Automatically rebalances using class weights
        scale_pos_weight=ratio  → Manual weight = n_negative / n_positive

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        n_estimators : Number of trees
        learning_rate: Shrinkage factor

    Returns:
        Dictionary comparing default vs imbalance-corrected models
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw   = n_neg / n_pos

    print(f"Class balance: 0={n_neg} | 1={n_pos} | "
          f"scale_pos_weight={spw:.2f}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15,
        stratify=y_train, random_state=42
    )
    callbacks = [lgb.early_stopping(30, verbose=False),
                 lgb.log_evaluation(period=-1)]

    results = {}
    configs = [
        ("Default",                  {"is_unbalance": False, "scale_pos_weight": 1.0}),
        ("is_unbalance=True",        {"is_unbalance": True,  "scale_pos_weight": 1.0}),
        (f"scale_pos_weight={spw:.1f}", {"is_unbalance": False, "scale_pos_weight": spw}),
    ]

    for name, extra_params in configs:
        model = LGBMClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, random_state=42,
            n_jobs=-1, verbose=-1, **extra_params
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=callbacks)

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
# 🔧 10. EVALUATION
# =============================================================================

def evaluate_lgbm_classifier(y_test: pd.Series,
                               y_pred: np.ndarray,
                               y_prob: np.ndarray = None,
                               model_name: str = "LightGBM") -> pd.DataFrame:
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

    # ── 1. LightGBM Classifier (sklearn API) ──────────────────────────────
    print("\n" + "=" * 65)
    print("1️⃣  LightGBM Classifier (sklearn API)")
    print("=" * 65)
    result = train_lgbm_classifier(X_train, X_test, y_train, y_test)
    evaluate_lgbm_classifier(y_test, result["y_pred"], result["y_prob"][:, 1])

    # ── 2. Feature Importance — Gain ───────────────────────────────────────
    print("\n" + "=" * 65)
    print("2️⃣  Feature Importance — Gain")
    print("=" * 65)
    imp_df = get_feature_importance(result["model"],
                                     X_train.columns.tolist(),
                                     importance_type="gain")

    # ── 3. Permutation Importance ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("3️⃣  Permutation Importance")
    print("=" * 65)
    perm_df = compute_permutation_importance(result["model"], X_test, y_test)

    # ── 4. num_leaves Sensitivity ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("4️⃣  num_leaves Sensitivity")
    print("=" * 65)
    nl_df = num_leaves_sensitivity(X_train, X_test, y_train, y_test)

    # ── 5. Native API ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("5️⃣  LightGBM Native API")
    print("=" * 65)
    native_result = train_lgbm_native(X_train, X_test, y_train, y_test)

    # ── 6. Class Imbalance ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("6️⃣  Class Imbalance Handling")
    print("=" * 65)
    imb_result = train_lgbm_imbalanced(X_train, X_test, y_train, y_test)

    # ── 7. RandomizedSearchCV ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("7️⃣  RandomizedSearchCV — Tuning")
    print("=" * 65)
    search_result = tune_lgbm(X_train, y_train, method="random", n_iter=20)

    # ── 8. Cross-Validation ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("8️⃣  Stratified 5-Fold CV")
    print("=" * 65)
    cv_result = cross_validate_lgbm(X, y)

    # ── 9. LightGBM Regressor ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("9️⃣  LightGBM Regressor")
    print("=" * 65)
    X_reg_raw, y_reg_raw = make_regression(
        n_samples=800, n_features=10, noise=20, random_state=42
    )
    X_reg = pd.DataFrame(X_reg_raw, columns=[f"F{i+1}" for i in range(10)])
    y_reg = pd.Series(y_reg_raw, name="Target")
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    reg_result = train_lgbm_regressor(Xr_tr, Xr_te, yr_tr, yr_te)

    print("\n✅ All LightGBM techniques demonstrated successfully!")
