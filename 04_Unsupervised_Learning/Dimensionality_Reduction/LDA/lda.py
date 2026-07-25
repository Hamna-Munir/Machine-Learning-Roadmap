# =============================================================================
# 📦 Linear Discriminant Analysis (LDA) — Reusable ML Script
# =============================================================================
# Author   : Hamna Munir
# Topic    : 04_Unsupervised_Learning / LDA
# File     : lda.py
# =============================================================================

# -----------------------------------------------------------------------------
# 📚 Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 🔧 1. FIT LDA — DIMENSIONALITY REDUCTION
# =============================================================================

def fit_lda(X: pd.DataFrame,
             y: pd.Series,
             n_components: int = None,
             solver: str = "svd",
             shrinkage=None,
             scale: bool = True) -> dict:
    """
    Fits LDA as a supervised dimensionality reduction technique.

    LDA finds the directions (Linear Discriminants) that maximize
    class separation — unlike PCA which maximizes total variance.

    Key formula:
        J(w) = wᵀSᴮw / wᵀSᵂw  (Fisher criterion)
        Sᴮ = between-class scatter
        Sᵂ = within-class scatter

    Maximum components: min(n_classes − 1, n_features)
        Binary → 1 discriminant axis (LD1 only)
        K classes → up to K−1 axes

    ⚠️ Always StandardScale before LDA — it uses distance-based
        scatter matrices sensitive to feature scale.

    Args:
        X            : Feature DataFrame
        y            : Target Series (class labels — REQUIRED)
        n_components : Number of discriminant axes to keep
                       (default: min(K−1, n_features))
        solver       : 'svd' (default, no cov matrix), 'lsqr', 'eigen'
        shrinkage    : None, 'auto' (Ledoit-Wolf), or float [0,1]
                       Use 'auto' when n_features > n_samples or
                       covariance matrix is singular
        scale        : Whether to StandardScale features (default: True)

    Returns:
        Dictionary with LDA model, transformed data, and stats
    """
    X_proc = X.copy()
    scaler = None

    if scale:
        scaler  = StandardScaler()
        X_proc  = pd.DataFrame(
            scaler.fit_transform(X), columns=X.columns
        )

    n_classes  = len(np.unique(y))
    max_comps  = min(n_classes - 1, X.shape[1])
    n_comp_use = n_components if n_components else max_comps

    lda = LinearDiscriminantAnalysis(
        n_components=n_comp_use,
        solver=solver,
        shrinkage=shrinkage,
    )
    X_lda = lda.fit_transform(X_proc, y)

    ld_cols  = [f"LD{i+1}" for i in range(n_comp_use)]
    X_lda_df = pd.DataFrame(X_lda, columns=ld_cols)

    evr     = lda.explained_variance_ratio_ if hasattr(
        lda, "explained_variance_ratio_") else None
    cum_evr = np.cumsum(evr) if evr is not None else None

    print(f"[LDA] n_components={n_comp_use} | solver={solver} | "
          f"shrinkage={shrinkage}")
    print(f"  Input shape    : {X.shape}")
    print(f"  Output shape   : {X_lda_df.shape}")
    print(f"  n_classes      : {n_classes} (max LDs = {max_comps})")
    if evr is not None:
        print(f"  Explained var  : "
              f"{[round(v*100,2) for v in evr]}%")
        print(f"  Cumulative var : "
              f"{[round(v*100,2) for v in cum_evr]}%")

    return {
        "lda"           : lda,
        "scaler"        : scaler,
        "X_lda"         : X_lda_df,
        "X_scaled"      : X_proc,
        "n_components"  : n_comp_use,
        "n_classes"     : n_classes,
        "explained_var" : evr,
        "cumulative_var": cum_evr,
        "class_means"   : pd.DataFrame(
            lda.means_, columns=X.columns,
            index=lda.classes_
        ),
    }


# =============================================================================
# 🔧 2. TRAIN LDA CLASSIFIER
# =============================================================================

def train_lda_classifier(X_train: pd.DataFrame,
                           X_test: pd.DataFrame,
                           y_train: pd.Series,
                           y_test: pd.Series,
                           solver: str = "svd",
                           shrinkage=None,
                           priors=None,
                           scale: bool = True) -> dict:
    """
    Trains LDA as a probabilistic classifier.

    LDA classifier uses Bayes' theorem with Gaussian assumptions:
        P(y=k|x) ∝ P(x|y=k) × P(y=k)
        Decision: argmax_k of the discriminant function

    Assumptions:
        - Features are Gaussian within each class
        - All classes share the SAME covariance matrix (homoscedastic)
        → If violated: use QDA (QuadraticDiscriminantAnalysis)

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        solver     : 'svd' (no cov matrix), 'lsqr', 'eigen'
        shrinkage  : Regularization — None, 'auto', or float [0,1]
                     Use 'auto' when n_features > n_samples
        priors     : Class prior probabilities (None = from data)
        scale      : Whether to StandardScale

    Returns:
        Dictionary with model, predictions, probabilities, and metrics
    """
    steps = [("scaler", StandardScaler())] if scale else []
    steps.append(("model", LinearDiscriminantAnalysis(
        solver=solver,
        shrinkage=shrinkage,
        priors=priors,
    )))

    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)

    model     = pipe.named_steps["model"]
    y_pred    = pipe.predict(X_test)
    y_prob    = pipe.predict_proba(X_test)
    n_classes = len(np.unique(y_train))

    metrics = _evaluate(y_test, y_pred,
                         y_prob[:, 1] if n_classes == 2 else None,
                         n_classes)

    print(f"[LDA Classifier] solver={solver} | shrinkage={shrinkage}")
    print(f"  Class priors (from data): "
          f"{dict(zip(model.classes_, model.priors_.round(4)))}")
    _print_metrics(metrics)

    return {
        "pipeline"  : pipe,
        "model"     : model,
        "y_pred"    : y_pred,
        "y_prob"    : y_prob,
        "metrics"   : metrics,
        "coef"      : pd.DataFrame(
            model.coef_, columns=X_train.columns,
            index=model.classes_
        ) if hasattr(model, "coef_") else None,
    }


# =============================================================================
# 🔧 3. TRAIN QDA CLASSIFIER
# =============================================================================

def train_qda_classifier(X_train: pd.DataFrame,
                           X_test: pd.DataFrame,
                           y_train: pd.Series,
                           y_test: pd.Series,
                           reg_param: float = 0.0,
                           scale: bool = True) -> dict:
    """
    Trains Quadratic Discriminant Analysis (QDA) classifier.

    Unlike LDA, QDA estimates a SEPARATE covariance matrix per class:
        → Quadratic decision boundary (more flexible)
        → Needs more data (more parameters to estimate)
        → Better when classes have genuinely different covariances

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        reg_param : Regularization [0, 1] — shrinks per-class covariances
                    toward a spherical estimate (0 = no reg, 1 = full reg)
        scale     : Whether to StandardScale

    Returns:
        Dictionary with model, predictions, and metrics
    """
    steps = [("scaler", StandardScaler())] if scale else []
    steps.append(("model", QuadraticDiscriminantAnalysis(
        reg_param=reg_param
    )))

    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)

    model     = pipe.named_steps["model"]
    y_pred    = pipe.predict(X_test)
    y_prob    = pipe.predict_proba(X_test)
    n_classes = len(np.unique(y_train))

    metrics = _evaluate(y_test, y_pred,
                         y_prob[:, 1] if n_classes == 2 else None,
                         n_classes)

    print(f"[QDA Classifier] reg_param={reg_param}")
    _print_metrics(metrics)

    return {
        "pipeline": pipe,
        "model"   : model,
        "y_pred"  : y_pred,
        "y_prob"  : y_prob,
        "metrics" : metrics,
    }


# =============================================================================
# 🔧 4. LDA vs QDA COMPARISON
# =============================================================================

def compare_lda_qda(X_train: pd.DataFrame,
                     X_test: pd.DataFrame,
                     y_train: pd.Series,
                     y_test: pd.Series,
                     scale: bool = True) -> pd.DataFrame:
    """
    Side-by-side comparison of LDA vs QDA on the same dataset.

    LDA: equal covariance assumption → linear boundary
    QDA: separate covariance per class → quadratic boundary

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        scale : Whether to StandardScale

    Returns:
        DataFrame comparing metrics for LDA and QDA
    """
    X_proc_tr = X_train.copy()
    X_proc_te = X_test.copy()

    if scale:
        sc       = StandardScaler()
        X_proc_tr = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns)
        X_proc_te = pd.DataFrame(sc.transform(X_test),  columns=X_test.columns)

    n_classes  = len(np.unique(y_train))
    rows       = []

    for name, model in [
        ("LDA (shrinkage=None)",  LinearDiscriminantAnalysis()),
        ("LDA (shrinkage=auto)",  LinearDiscriminantAnalysis(
            shrinkage="auto", solver="lsqr")),
        ("QDA (reg=0.0)",         QuadraticDiscriminantAnalysis()),
        ("QDA (reg=0.1)",         QuadraticDiscriminantAnalysis(reg_param=0.1)),
    ]:
        model.fit(X_proc_tr, y_train)
        y_pred = model.predict(X_proc_te)
        y_prob = model.predict_proba(X_proc_te)
        tr_acc = accuracy_score(y_train, model.predict(X_proc_tr))
        te_acc = accuracy_score(y_test, y_pred)
        te_f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        row = {
            "Model"    : name,
            "Train Acc": round(tr_acc, 4),
            "Test Acc" : round(te_acc, 4),
            "Test F1"  : round(te_f1, 4),
        }
        if n_classes == 2:
            row["ROC-AUC"] = round(roc_auc_score(y_test, y_prob[:, 1]), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    print("LDA vs QDA Comparison:")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 5. LDA vs PCA COMPARISON
# =============================================================================

def compare_lda_pca(X_train: pd.DataFrame,
                     X_test: pd.DataFrame,
                     y_train: pd.Series,
                     y_test: pd.Series,
                     estimator,
                     n_components: int = 2,
                     scale: bool = True) -> pd.DataFrame:
    """
    Compares classification performance after LDA vs PCA reduction.

    LDA (supervised)   → maximizes class separation
    PCA (unsupervised) → maximizes total variance

    Both projected to n_components, then same classifier applied.

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        estimator    : sklearn classifier to evaluate post-reduction
        n_components : Number of components for both LDA and PCA
        scale        : Whether to StandardScale

    Returns:
        DataFrame comparing LDA vs PCA on classifier performance
    """
    import copy

    n_classes  = len(np.unique(y_train))
    max_lda    = min(n_classes - 1, X_train.shape[1])
    n_lda      = min(n_components, max_lda)

    rows = []
    configs = [
        ("No Reduction",   None,   None),
        (f"PCA(k={n_components})",
         PCA(n_components=n_components, random_state=42), None),
        (f"LDA(k={n_lda})",
         LinearDiscriminantAnalysis(n_components=n_lda),
         y_train),
    ]

    for name, reducer, y_fit in configs:
        steps = [("scaler", StandardScaler())]
        if reducer is not None:
            steps.append(("reducer", reducer))
        steps.append(("model", copy.deepcopy(estimator)))
        pipe = Pipeline(steps)

        if y_fit is not None:
            # LDA reducer needs y during fit — handled inside Pipeline
            pipe.fit(X_train, y_train)
        else:
            pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test) if hasattr(estimator, "predict_proba") else None
        te_acc = accuracy_score(y_test, y_pred)
        te_f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        row = {
            "Method"  : name,
            "Test Acc": round(te_acc, 4),
            "Test F1" : round(te_f1, 4),
        }
        if y_prob is not None and n_classes == 2:
            row["ROC-AUC"] = round(roc_auc_score(y_test, y_prob[:, 1]), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"LDA vs PCA Reduction Comparison "
          f"(estimator: {estimator.__class__.__name__}):")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 6. SHRINKAGE SENSITIVITY
# =============================================================================

def shrinkage_sensitivity(X_train: pd.DataFrame,
                            X_test: pd.DataFrame,
                            y_train: pd.Series,
                            y_test: pd.Series,
                            shrinkage_values: list = None,
                            scale: bool = True) -> pd.DataFrame:
    """
    Evaluates LDA performance across a range of shrinkage values.

    Shrinkage regularizes the covariance matrix estimate:
        α = 0   → No regularization (standard sample covariance)
        α = 1   → Fully shrunk (spherical covariance)
        α='auto' → Ledoit-Wolf optimal shrinkage

    Needed when n_features ≥ n_samples or covariance is near-singular.

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        shrinkage_values : List of alpha values to evaluate
        scale            : Whether to StandardScale

    Returns:
        DataFrame with metrics per shrinkage value
    """
    if shrinkage_values is None:
        shrinkage_values = [None, "auto", 0.0, 0.1, 0.2, 0.3,
                             0.5, 0.7, 0.9, 1.0]

    n_classes = len(np.unique(y_train))
    X_proc_tr = X_train.copy()
    X_proc_te = X_test.copy()

    if scale:
        sc        = StandardScaler()
        X_proc_tr = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns)
        X_proc_te = pd.DataFrame(sc.transform(X_test),  columns=X_test.columns)

    rows = []
    for sh in shrinkage_values:
        # shrinkage requires lsqr or eigen solver (not svd)
        solver = "svd" if sh is None else "lsqr"
        model  = LinearDiscriminantAnalysis(shrinkage=sh, solver=solver)
        try:
            model.fit(X_proc_tr, y_train)
            y_pred = model.predict(X_proc_te)
            y_prob = model.predict_proba(X_proc_te)
            te_acc = accuracy_score(y_test, y_pred)
            te_f1  = f1_score(y_test, y_pred,
                               average="weighted", zero_division=0)
            row = {
                "Shrinkage": str(sh),
                "Test Acc" : round(te_acc, 4),
                "Test F1"  : round(te_f1, 4),
            }
            if n_classes == 2:
                row["ROC-AUC"] = round(roc_auc_score(y_test, y_prob[:,1]), 4)
            rows.append(row)
        except Exception as e:
            rows.append({"Shrinkage": str(sh), "Error": str(e)})

    df = pd.DataFrame(rows)
    print("Shrinkage Sensitivity Analysis (LDA):")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 7. CLASS SEPARATION ANALYSIS
# =============================================================================

def class_separation_analysis(X: pd.DataFrame,
                                y: pd.Series,
                                scale: bool = True) -> pd.DataFrame:
    """
    Computes within-class and between-class scatter statistics —
    quantifies how well-separated the classes are in feature space.

    Higher between/within ratio → easier to discriminate classes.

    Args:
        X     : Feature DataFrame
        y     : Target Series (class labels)
        scale : Whether to StandardScale

    Returns:
        DataFrame with scatter statistics per feature
    """
    X_proc = X.copy()
    if scale:
        X_proc = pd.DataFrame(
            StandardScaler().fit_transform(X), columns=X.columns
        )

    classes     = np.unique(y)
    global_mean = X_proc.mean()

    within_var  = pd.Series(0.0, index=X.columns)
    between_var = pd.Series(0.0, index=X.columns)

    for cls in classes:
        mask      = y.values == cls
        cls_data  = X_proc[mask]
        cls_mean  = cls_data.mean()
        nk        = mask.sum()
        within_var  += cls_data.var() * (nk - 1)
        between_var += nk * (cls_mean - global_mean) ** 2

    rows = []
    for feat in X.columns:
        wv = within_var[feat]
        bv = between_var[feat]
        rows.append({
            "Feature"         : feat,
            "Within Scatter"  : round(wv, 4),
            "Between Scatter" : round(bv, 4),
            "B/W Ratio"       : round(bv / (wv + 1e-10), 4),
        })

    df = pd.DataFrame(rows).sort_values("B/W Ratio", ascending=False
                                         ).reset_index(drop=True)
    print(f"[Class Separation Analysis] "
          f"{len(classes)} classes | {len(X)} samples")
    print(df.head(10).to_string(index=False))
    print(f"\n  Top features for separation: "
          f"{df['Feature'].head(5).tolist()}")
    return df


# =============================================================================
# 🔧 8. CROSS-VALIDATION
# =============================================================================

def cross_validate_lda(X: pd.DataFrame,
                         y: pd.Series,
                         solver: str = "svd",
                         shrinkage=None,
                         cv: int = 5,
                         scoring: str = "accuracy",
                         scale: bool = True) -> dict:
    """
    Performs Stratified K-Fold Cross-Validation on LDA Pipeline.

    Args:
        X         : Full feature DataFrame
        y         : Full target Series
        solver    : LDA solver
        shrinkage : Regularization parameter
        cv        : Number of folds
        scoring   : Scoring metric
        scale     : Whether to StandardScale

    Returns:
        Dictionary with fold scores, mean, and std
    """
    steps = [("scaler", StandardScaler())] if scale else []
    steps.append(("model", LinearDiscriminantAnalysis(
        solver=solver, shrinkage=shrinkage
    )))
    pipe   = Pipeline(steps)
    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=skf, scoring=scoring)

    result = {
        "scores"      : scores,
        "mean"        : scores.mean(),
        "std"         : scores.std(),
        "fold_results": {f"Fold {i+1}": round(s, 4) for i, s in enumerate(scores)},
    }

    print(f"[LDA CV] solver={solver} | shrinkage={shrinkage} | "
          f"cv={cv} | {scoring}: {scores.mean():.4f} ± {scores.std():.4f}")
    for fold, score in result["fold_results"].items():
        print(f"  {fold}: {score}")

    return result


# =============================================================================
# 🔧 9. LDA FOR VISUALIZATION (2D / 3D)
# =============================================================================

def lda_for_visualization(X: pd.DataFrame,
                            y: pd.Series,
                            n_components: int = 2,
                            scale: bool = True) -> pd.DataFrame:
    """
    Projects data to 2D or 3D using LDA for supervised visualization.

    Unlike PCA (variance-based), LDA projection maximally separates
    the class distributions — making it ideal for visualizing whether
    classes are distinguishable in the data.

    Args:
        X            : Feature DataFrame
        y            : Target Series (class labels)
        n_components : 2 (2D plot) or 3 (3D plot)
        scale        : Whether to StandardScale

    Returns:
        DataFrame with LD1, LD2 (optionally LD3), and Label columns
    """
    n_classes  = len(np.unique(y))
    max_comps  = min(n_classes - 1, X.shape[1])
    n_comp_use = min(n_components, max_comps)

    if n_comp_use < n_components:
        print(f"  ⚠️ Requested {n_components} components but max for "
              f"{n_classes} classes = {max_comps}. Using {n_comp_use}.")

    X_proc = X.copy()
    if scale:
        X_proc = pd.DataFrame(
            StandardScaler().fit_transform(X), columns=X.columns
        )

    lda   = LinearDiscriminantAnalysis(n_components=n_comp_use)
    X_lda = lda.fit_transform(X_proc, y)
    evr   = lda.explained_variance_ratio_

    ld_cols   = [f"LD{i+1}" for i in range(n_comp_use)]
    result_df = pd.DataFrame(X_lda, columns=ld_cols)
    result_df["Label"] = y.values if hasattr(y, "values") else y

    print(f"[LDA Visualization] {X.shape[1]}D → {n_comp_use}D")
    for col, var in zip(ld_cols, evr):
        print(f"  {col}: {var*100:.2f}% class separation explained")

    return result_df


# =============================================================================
# 🔧 10. EVALUATION
# =============================================================================

def evaluate_lda_classifier(y_test: pd.Series,
                              y_pred: np.ndarray,
                              y_prob: np.ndarray = None,
                              model_name: str = "LDA") -> pd.DataFrame:
    """Computes and displays a full classification evaluation report."""
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
    from sklearn.linear_model import LogisticRegression

    np.random.seed(42)

    # ── Binary classification dataset ─────────────────────────────────────
    X_raw, y_raw = make_classification(
        n_samples=800, n_features=20, n_informative=8,
        n_redundant=6, n_classes=2, weights=[0.6, 0.4],
        random_state=42
    )
    X = pd.DataFrame(X_raw, columns=[f"Feature_{i+1:02d}" for i in range(20)])
    y = pd.Series(y_raw, name="Target")

    # ── Multiclass dataset ─────────────────────────────────────────────────
    X_mc_raw, y_mc_raw = make_classification(
        n_samples=600, n_features=10, n_informative=6,
        n_redundant=2, n_classes=4, random_state=42
    )
    X_mc = pd.DataFrame(X_mc_raw,
                         columns=[f"F_{i+1}" for i in range(10)])
    y_mc = pd.Series(y_mc_raw, name="Class")

    print("=" * 65)
    print("📊 Dataset 1 — Binary (20 features)")
    print("=" * 65)
    print(f"Shape: {X.shape} | Classes: {dict(y.value_counts().sort_index())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── 1. LDA Dimensionality Reduction ───────────────────────────────────
    print("\n" + "=" * 65)
    print("1️⃣  LDA — Dimensionality Reduction")
    print("=" * 65)
    lda_result = fit_lda(X_train, y_train)

    # ── 2. LDA Classifier ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("2️⃣  LDA Classifier")
    print("=" * 65)
    clf_result = train_lda_classifier(X_train, X_test, y_train, y_test)
    evaluate_lda_classifier(y_test, clf_result["y_pred"],
                              clf_result["y_prob"][:, 1])

    # ── 3. QDA Classifier ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("3️⃣  QDA Classifier")
    print("=" * 65)
    qda_result = train_qda_classifier(X_train, X_test, y_train, y_test)

    # ── 4. LDA vs QDA Comparison ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("4️⃣  LDA vs QDA Comparison")
    print("=" * 65)
    comp_df = compare_lda_qda(X_train, X_test, y_train, y_test)

    # ── 5. Shrinkage Sensitivity ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("5️⃣  Shrinkage Sensitivity")
    print("=" * 65)
    sh_df = shrinkage_sensitivity(X_train, X_test, y_train, y_test)

    # ── 6. Class Separation Analysis ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("6️⃣  Class Separation Analysis")
    print("=" * 65)
    sep_df = class_separation_analysis(X_train, y_train)

    # ── 7. LDA vs PCA Comparison ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("7️⃣  LDA vs PCA Reduction Comparison")
    print("=" * 65)
    lda_pca_df = compare_lda_pca(
        X_train, X_test, y_train, y_test,
        estimator=LogisticRegression(max_iter=1000, random_state=42),
        n_components=1
    )

    # ── 8. Cross-Validation ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("8️⃣  Stratified 5-Fold CV")
    print("=" * 65)
    cv_result = cross_validate_lda(X, y, scoring="roc_auc")

    # ── 9. Multiclass LDA ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("9️⃣  Multiclass LDA (4 classes → max 3 LDs)")
    print("=" * 65)
    Xmc_tr, Xmc_te, ymc_tr, ymc_te = train_test_split(
        X_mc, y_mc, test_size=0.2, stratify=y_mc, random_state=42
    )
    mc_result = train_lda_classifier(Xmc_tr, Xmc_te, ymc_tr, ymc_te)

    # ── 10. LDA for 2D Visualization ─────────────────────────────────────
    print("\n" + "=" * 65)
    print("🔟  LDA for 2D Visualization (multiclass)")
    print("=" * 65)
    viz_df = lda_for_visualization(X_mc, y_mc, n_components=2)
    print(viz_df.head())

    print("\n✅ All LDA techniques demonstrated successfully!")
