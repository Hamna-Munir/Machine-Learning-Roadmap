# =============================================================================
# 📦 Naive Bayes — Reusable ML Script
# =============================================================================
# Author   : Hamna Munir
# Topic    : 03_Supervised_Learning / Classification / Naive_Bayes
# File     : naive_bayes.py
# =============================================================================

# -----------------------------------------------------------------------------
# 📚 Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss,
    confusion_matrix, classification_report
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 🔧 1. TRAIN GAUSSIAN NAIVE BAYES (Continuous Features)
# =============================================================================

def train_gaussian_nb(X_train: pd.DataFrame,
                       X_test: pd.DataFrame,
                       y_train: pd.Series,
                       y_test: pd.Series,
                       var_smoothing: float = 1e-9,
                       scale: bool = False) -> dict:
    """
    Trains a Gaussian Naive Bayes classifier for continuous features.

    Formula:
        P(xᵢ|y) = Gaussian(xᵢ; μ_iy, σ²_iy)

    Best for:
        - Continuous numerical features
        - Features that are approximately normally distributed within each class

    Note:
        GaussianNB does NOT require scaling — it estimates its own mean/variance
        per feature per class. Scaling does not change predictions mathematically,
        though it can help with var_smoothing numerical stability.

    Args:
        X_train       : Training features DataFrame
        X_test        : Test features DataFrame
        y_train       : Training target Series
        y_test        : Test target Series
        var_smoothing : Portion of largest variance added for stability
        scale         : Whether to StandardScale (default: False, not required)

    Returns:
        Dictionary with model, predictions, probabilities, and metrics
    """
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", GaussianNB(var_smoothing=var_smoothing)))

    pipe = Pipeline(steps)
    pipe.fit(X_train, y_train)

    model     = pipe.named_steps["model"]
    y_pred    = pipe.predict(X_test)
    y_prob    = pipe.predict_proba(X_test)
    n_classes = len(np.unique(y_train))

    metrics = _evaluate(y_test, y_pred,
                         y_prob[:, 1] if n_classes == 2 else None,
                         n_classes)

    print(f"[GaussianNB] var_smoothing={var_smoothing}")
    print(f"  Class priors: {dict(zip(model.classes_, model.class_prior_.round(4)))}")
    _print_metrics(metrics)

    return {
        "pipeline" : pipe,
        "model"    : model,
        "y_pred"   : y_pred,
        "y_prob"   : y_prob,
        "metrics"  : metrics,
        "means"    : pd.DataFrame(model.theta_, columns=X_train.columns,
                                   index=model.classes_),
        "variances": pd.DataFrame(model.var_, columns=X_train.columns,
                                   index=model.classes_),
    }


# =============================================================================
# 🔧 2. TRAIN MULTINOMIAL NAIVE BAYES (Count Data / Text)
# =============================================================================

def train_multinomial_nb(X_train,
                          X_test,
                          y_train: pd.Series,
                          y_test: pd.Series,
                          alpha: float = 1.0) -> dict:
    """
    Trains a Multinomial Naive Bayes classifier for count-based features.

    Formula:
        P(xᵢ|y) = (count(xᵢ,y) + α) / (count(y) + α × n_features)

    Best for:
        - Text classification (word counts, TF-IDF)
        - Any non-negative integer count features

    Args:
        X_train : Training feature matrix (counts, non-negative)
        X_test  : Test feature matrix
        y_train : Training target Series
        y_test  : Test target Series
        alpha   : Laplace smoothing parameter (default: 1.0)

    Returns:
        Dictionary with model, predictions, probabilities, and metrics
    """
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)

    y_pred    = model.predict(X_test)
    y_prob    = model.predict_proba(X_test)
    n_classes = len(np.unique(y_train))

    metrics = _evaluate(y_test, y_pred,
                         y_prob[:, 1] if n_classes == 2 else None,
                         n_classes)

    print(f"[MultinomialNB] alpha={alpha}")
    print(f"  Class priors: {dict(zip(model.classes_, np.exp(model.class_log_prior_).round(4)))}")
    _print_metrics(metrics)

    return {
        "model"  : model,
        "y_pred" : y_pred,
        "y_prob" : y_prob,
        "metrics": metrics,
    }


# =============================================================================
# 🔧 3. TRAIN BERNOULLI NAIVE BAYES (Binary Features)
# =============================================================================

def train_bernoulli_nb(X_train,
                        X_test,
                        y_train: pd.Series,
                        y_test: pd.Series,
                        alpha: float = 1.0,
                        binarize: float = 0.0) -> dict:
    """
    Trains a Bernoulli Naive Bayes classifier for binary features.

    Formula:
        P(xᵢ|y) = P(xᵢ=1|y)^xᵢ × (1−P(xᵢ=1|y))^(1−xᵢ)

    Best for:
        - Binary presence/absence features
        - Short text documents (word presence vs frequency)

    Args:
        X_train  : Training feature matrix
        X_test   : Test feature matrix
        y_train  : Training target Series
        y_test   : Test target Series
        alpha    : Laplace smoothing parameter (default: 1.0)
        binarize : Threshold to convert continuous to binary (default: 0.0)

    Returns:
        Dictionary with model, predictions, probabilities, and metrics
    """
    model = BernoulliNB(alpha=alpha, binarize=binarize)
    model.fit(X_train, y_train)

    y_pred    = model.predict(X_test)
    y_prob    = model.predict_proba(X_test)
    n_classes = len(np.unique(y_train))

    metrics = _evaluate(y_test, y_pred,
                         y_prob[:, 1] if n_classes == 2 else None,
                         n_classes)

    print(f"[BernoulliNB] alpha={alpha} | binarize={binarize}")
    _print_metrics(metrics)

    return {
        "model"  : model,
        "y_pred" : y_pred,
        "y_prob" : y_prob,
        "metrics": metrics,
    }


# =============================================================================
# 🔧 4. TEXT CLASSIFICATION PIPELINE (TF-IDF + MultinomialNB)
# =============================================================================

def train_text_classifier(texts_train: list,
                           texts_test: list,
                           y_train: pd.Series,
                           y_test: pd.Series,
                           vectorizer: str = "tfidf",
                           alpha: float = 1.0,
                           max_features: int = 5000,
                           ngram_range: tuple = (1, 1)) -> dict:
    """
    Builds a full text classification pipeline using Naive Bayes.

    Pipeline: Raw text → Vectorizer (Count/TF-IDF) → MultinomialNB

    Args:
        texts_train  : List/Series of training text documents
        texts_test   : List/Series of test text documents
        y_train      : Training labels
        y_test       : Test labels
        vectorizer   : 'count' or 'tfidf' (default: 'tfidf')
        alpha        : Laplace smoothing for Naive Bayes
        max_features : Maximum vocabulary size
        ngram_range  : N-gram range, e.g., (1,1) unigrams, (1,2) uni+bigrams

    Returns:
        Dictionary with pipeline, predictions, and metrics
    """
    if vectorizer == "tfidf":
        vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range,
                               stop_words="english")
    else:
        vec = CountVectorizer(max_features=max_features, ngram_range=ngram_range,
                               stop_words="english")

    pipe = Pipeline([
        ("vectorizer", vec),
        ("model", MultinomialNB(alpha=alpha))
    ])
    pipe.fit(texts_train, y_train)

    y_pred = pipe.predict(texts_test)
    y_prob = pipe.predict_proba(texts_test)
    n_classes = len(np.unique(y_train))

    metrics = _evaluate(y_test, y_pred,
                         y_prob[:, 1] if n_classes == 2 else None,
                         n_classes)

    print(f"[Text Classifier] vectorizer={vectorizer} | alpha={alpha} | "
          f"vocab_size={len(pipe.named_steps['vectorizer'].vocabulary_)}")
    _print_metrics(metrics)

    # Top informative words per class (for binary)
    if n_classes == 2:
        feature_names = pipe.named_steps["vectorizer"].get_feature_names_out()
        log_probs = pipe.named_steps["model"].feature_log_prob_
        top_class1 = pd.Series(log_probs[1], index=feature_names).nlargest(10)
        top_class0 = pd.Series(log_probs[0], index=feature_names).nlargest(10)
        print(f"\n  Top words for class {pipe.named_steps['model'].classes_[1]}: "
              f"{list(top_class1.index)}")
        print(f"  Top words for class {pipe.named_steps['model'].classes_[0]}: "
              f"{list(top_class0.index)}")

    return {
        "pipeline": pipe,
        "y_pred"  : y_pred,
        "y_prob"  : y_prob,
        "metrics" : metrics,
    }


# =============================================================================
# 🔧 5. COMPARE ALL NB VARIANTS
# =============================================================================

def compare_nb_variants(X_train: pd.DataFrame,
                         X_test: pd.DataFrame,
                         y_train: pd.Series,
                         y_test: pd.Series) -> pd.DataFrame:
    """
    Compares Gaussian, Multinomial, Bernoulli, and Complement Naive Bayes
    on the same dataset (after appropriate scaling for each variant).

    Note: Multinomial requires non-negative features — uses MinMax-scaled
    version internally for fair comparison.

    Args:
        X_train : Training features DataFrame
        X_test  : Test features DataFrame
        y_train : Training target Series
        y_test  : Test target Series

    Returns:
        DataFrame with comparison metrics across all NB variants
    """
    # Non-negative version for Multinomial/Complement/Bernoulli
    mm_scaler = MinMaxScaler()
    X_tr_mm   = mm_scaler.fit_transform(X_train)
    X_te_mm   = mm_scaler.transform(X_test)

    n_classes = len(np.unique(y_train))

    models = {
        "GaussianNB"    : (GaussianNB(), X_train, X_test),
        "MultinomialNB" : (MultinomialNB(), X_tr_mm, X_te_mm),
        "BernoulliNB"   : (BernoulliNB(binarize=0.5), X_tr_mm, X_te_mm),
        "ComplementNB"  : (ComplementNB(), X_tr_mm, X_te_mm),
    }

    rows = []
    for name, (model, X_tr, X_te) in models.items():
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)

        row = {
            "Model"    : name,
            "Accuracy" : round(accuracy_score(y_test, y_pred), 4),
            "F1"       : round(f1_score(y_test, y_pred,
                                        average="weighted", zero_division=0), 4),
        }
        if n_classes == 2:
            row["ROC-AUC"] = round(roc_auc_score(y_test, y_prob[:, 1]), 4)
            row["LogLoss"] = round(log_loss(y_test, y_prob[:, 1]), 4)
        rows.append(row)

    comp_df = pd.DataFrame(rows)
    print("Naive Bayes Variant Comparison:")
    print(comp_df.to_string(index=False))
    return comp_df


# =============================================================================
# 🔧 6. ALPHA SENSITIVITY (Laplace Smoothing)
# =============================================================================

def alpha_sensitivity(X_train,
                       X_test,
                       y_train: pd.Series,
                       y_test: pd.Series,
                       model_type: str = "multinomial",
                       alphas: list = None) -> pd.DataFrame:
    """
    Evaluates Naive Bayes performance across a range of alpha
    (Laplace smoothing) values.

    Args:
        X_train    : Training features
        X_test     : Test features
        y_train    : Training labels
        y_test     : Test labels
        model_type : 'multinomial' or 'bernoulli'
        alphas     : List of alpha values to evaluate

    Returns:
        DataFrame with metrics per alpha value
    """
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    rows = []
    for a in alphas:
        if model_type == "multinomial":
            model = MultinomialNB(alpha=a)
        else:
            model = BernoulliNB(alpha=a)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        rows.append({
            "Alpha"   : a,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "F1"      : round(f1_score(y_test, y_pred,
                                       average="weighted", zero_division=0), 4),
        })

    df = pd.DataFrame(rows)
    print(f"Alpha Sensitivity ({model_type}):")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 7. GRIDSEARCHCV — NAIVE BAYES TUNING
# =============================================================================

def tune_naive_bayes(X_train,
                      y_train: pd.Series,
                      model_type: str = "gaussian",
                      cv: int = 5,
                      scoring: str = "f1_weighted") -> dict:
    """
    Tunes Naive Bayes hyperparameters using GridSearchCV.

    Args:
        X_train    : Training features
        y_train    : Training target
        model_type : 'gaussian', 'multinomial', or 'bernoulli'
        cv         : Number of CV folds
        scoring    : Scoring metric

    Returns:
        Dictionary with best params, score, and GridSearchCV object
    """
    if model_type == "gaussian":
        model = GaussianNB()
        param_grid = {"var_smoothing": np.logspace(-12, -1, 12).tolist()}
    elif model_type == "multinomial":
        model = MultinomialNB()
        param_grid = {"alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]}
    else:
        model = BernoulliNB()
        param_grid = {
            "alpha"   : [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
            "binarize": [0.0, 0.25, 0.5, 0.75],
        }

    grid = GridSearchCV(
        model, param_grid,
        cv=cv, scoring=scoring, n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print(f"[GridSearchCV {model_type.title()}NB] Best params: {grid.best_params_}")
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

def cross_validate_nb(X,
                       y: pd.Series,
                       model_type: str = "gaussian",
                       cv: int = 5,
                       scoring: str = "f1_weighted",
                       **model_params) -> dict:
    """
    Performs Stratified K-Fold Cross-Validation on a Naive Bayes model.

    Args:
        X          : Full feature matrix
        y          : Full target Series
        model_type : 'gaussian', 'multinomial', or 'bernoulli'
        cv         : Number of folds
        scoring    : Scoring metric
        **model_params : Additional model hyperparameters

    Returns:
        Dictionary with fold scores, mean, and std
    """
    model_map = {
        "gaussian"   : GaussianNB,
        "multinomial": MultinomialNB,
        "bernoulli"  : BernoulliNB,
    }
    model = model_map[model_type](**model_params)
    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)

    result = {
        "scores"      : scores,
        "mean"        : scores.mean(),
        "std"         : scores.std(),
        "fold_results": {f"Fold {i+1}": round(s, 4) for i, s in enumerate(scores)},
    }

    print(f"[{model_type.title()}NB CV] cv={cv} | "
          f"{scoring}: {scores.mean():.4f} ± {scores.std():.4f}")
    for fold, score in result["fold_results"].items():
        print(f"  {fold}: {score}")

    return result


# =============================================================================
# 🔧 9. EVALUATION REPORT
# =============================================================================

def evaluate_nb_classifier(y_test: pd.Series,
                            y_pred: np.ndarray,
                            y_prob: np.ndarray = None,
                            model_name: str = "Naive Bayes") -> pd.DataFrame:
    """
    Computes and displays a full classification evaluation report.

    Args:
        y_test     : True labels
        y_pred     : Predicted labels
        y_prob     : Predicted probabilities for class 1 (optional)
        model_name : Name for display

    Returns:
        DataFrame with evaluation metrics
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


def _print_metrics(metrics: dict) -> None:
    m = metrics.get("test", {})
    print(f"  [TEST]  Acc={m.get('Accuracy',0):.4f} | "
          f"F1={m.get('F1',0):.4f} | "
          f"AUC={m.get('ROC-AUC','N/A')}")


# =============================================================================
# 🚀 MAIN — Demo with Synthetic Datasets
# =============================================================================

if __name__ == "__main__":

    from sklearn.datasets import make_classification

    np.random.seed(42)

    # ── 1. Gaussian NB Demo — Continuous Features ──────────────────────────
    print("=" * 65)
    print("📊 DEMO 1 — Gaussian Naive Bayes (Continuous Features)")
    print("=" * 65)

    X_raw, y_raw = make_classification(
        n_samples=800, n_features=8, n_informative=5,
        n_redundant=1, n_classes=2, weights=[0.65, 0.35],
        random_state=42
    )
    X = pd.DataFrame(X_raw, columns=[f"Feat_{i+1}" for i in range(8)])
    y = pd.Series(y_raw, name="Target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    result_gnb = train_gaussian_nb(X_train, X_test, y_train, y_test)
    evaluate_nb_classifier(y_test, result_gnb["y_pred"], result_gnb["y_prob"][:, 1])

    # ── 2. Compare all NB variants ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("📊 DEMO 2 — Compare All NB Variants")
    print("=" * 65)
    compare_df = compare_nb_variants(X_train, X_test, y_train, y_test)

    # ── 3. GridSearchCV for GaussianNB ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("📊 DEMO 3 — GridSearchCV — GaussianNB var_smoothing")
    print("=" * 65)
    gs_result = tune_naive_bayes(X_train, y_train, model_type="gaussian")

    # ── 4. Cross-Validation ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("📊 DEMO 4 — Stratified 5-Fold CV")
    print("=" * 65)
    cv_result = cross_validate_nb(X, y, model_type="gaussian", cv=5)

    # ── 5. Text Classification Demo ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("📊 DEMO 5 — Text Classification (TF-IDF + MultinomialNB)")
    print("=" * 65)

    texts = [
        "free money winner claim now urgent",
        "meeting scheduled tomorrow at noon",
        "win cash prize click here now",
        "project deadline extended please review",
        "congratulations you won lottery claim",
        "team standup meeting notes attached",
        "limited offer buy now discount",
        "please find attached quarterly report",
        "act now exclusive deal expires today",
        "lunch plans this friday with team",
    ] * 10  # repeat for more samples

    labels = ([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]) * 10  # 1=spam, 0=not spam

    texts_train, texts_test, y_text_train, y_text_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    text_result = train_text_classifier(
        texts_train, texts_test,
        pd.Series(y_text_train), pd.Series(y_text_test),
        vectorizer="tfidf", alpha=1.0
    )

    print("\n✅ All Naive Bayes techniques demonstrated successfully!")
