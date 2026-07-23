# =============================================================================
# 📦 Principal Component Analysis (PCA) — Reusable ML Script
# =============================================================================
# Author   : Hamna Munir
# Topic    : 04_Unsupervised_Learning / PCA
# File     : pca.py
# =============================================================================

# -----------------------------------------------------------------------------
# 📚 Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 🔧 1. FIT PCA
# =============================================================================

def fit_pca(X: pd.DataFrame,
             n_components=None,
             scale: bool = True,
             whiten: bool = False,
             random_state: int = 42) -> dict:
    """
    Fits PCA on a feature DataFrame.

    Algorithm:
        1. Center data (subtract mean)
        2. Compute covariance matrix C = XᵀX / (n−1)
        3. Eigendecomposition: C = V Λ Vᵀ
        4. Sort eigenvectors by eigenvalue (descending)
        5. Project data: X_reduced = X_centered × Vₖ

    ⚠️ Always StandardScale before PCA — high-variance features
        will dominate the principal components without scaling.

    Args:
        X            : Feature DataFrame
        n_components : Number of components to keep
                       int   → exact number of PCs
                       float → fraction of variance to retain (e.g., 0.95)
                       None  → keep all components
                       'mle' → Minka's MLE for auto-selection
        scale        : Whether to StandardScale features (default: True)
        whiten       : Scale each PC to unit variance (default: False)
        random_state : Reproducibility seed

    Returns:
        Dictionary with pca model, scaler, transformed data, and stats
    """
    X_proc = X.copy()
    scaler = None

    if scale:
        scaler  = StandardScaler()
        X_proc  = pd.DataFrame(
            scaler.fit_transform(X), columns=X.columns
        )

    pca = PCA(
        n_components=n_components,
        whiten=whiten,
        svd_solver="randomized" if (
            isinstance(n_components, int) and n_components < X.shape[1] * 0.8
        ) else "full",
        random_state=random_state,
    )
    X_pca = pca.fit_transform(X_proc)

    n_comp_actual = pca.n_components_
    evr           = pca.explained_variance_ratio_
    cumulative_evr = np.cumsum(evr)

    pc_cols  = [f"PC{i+1}" for i in range(n_comp_actual)]
    X_pca_df = pd.DataFrame(X_pca, columns=pc_cols)

    # Loadings = eigenvectors (principal axes)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=pc_cols
    )

    print(f"[PCA] n_components={n_components} → actual={n_comp_actual} | "
          f"scale={scale}")
    print(f"  Input shape    : {X.shape}")
    print(f"  Output shape   : {X_pca_df.shape}")
    print(f"  Explained var  : "
          f"{[round(v*100,2) for v in evr[:5]]}% ...")
    print(f"  Cumulative var : "
          f"{[round(v*100,2) for v in cumulative_evr[:5]]}% ...")
    if cumulative_evr[-1] < 1.0:
        for k, cv in enumerate(cumulative_evr):
            if cv >= 0.90:
                print(f"  90% variance   : {k+1} components")
                break
        for k, cv in enumerate(cumulative_evr):
            if cv >= 0.95:
                print(f"  95% variance   : {k+1} components")
                break

    return {
        "pca"             : pca,
        "scaler"          : scaler,
        "X_pca"           : X_pca_df,
        "X_scaled"        : X_proc,
        "n_components"    : n_comp_actual,
        "explained_var"   : evr,
        "cumulative_var"  : cumulative_evr,
        "eigenvalues"     : pca.explained_variance_,
        "loadings"        : loadings,
        "feature_names"   : X.columns.tolist(),
    }


# =============================================================================
# 🔧 2. TRANSFORM NEW DATA
# =============================================================================

def transform_pca(pca: PCA,
                   scaler: StandardScaler,
                   X_new: pd.DataFrame) -> pd.DataFrame:
    """
    Projects new data onto the fitted PCA components.

    ⚠️ Always use the scaler and PCA fitted on TRAINING data.
        Never refit on test/new data — that would cause data leakage.

    Args:
        pca    : Fitted PCA object
        scaler : Fitted StandardScaler (or None)
        X_new  : New feature DataFrame

    Returns:
        DataFrame of projected PCA scores
    """
    X_proc = X_new.copy()
    if scaler is not None:
        X_proc = pd.DataFrame(
            scaler.transform(X_new), columns=X_new.columns
        )

    X_pca   = pca.transform(X_proc)
    pc_cols = [f"PC{i+1}" for i in range(pca.n_components_)]
    return pd.DataFrame(X_pca, columns=pc_cols)


# =============================================================================
# 🔧 3. RECONSTRUCTION & RECONSTRUCTION ERROR
# =============================================================================

def reconstruction_analysis(X: pd.DataFrame,
                              k_values: list = None,
                              scale: bool = True) -> pd.DataFrame:
    """
    Computes reconstruction error for different numbers of components.

    PCA reconstructs original data from k components:
        X_reconstructed ≈ X_pca @ Vₖᵀ + mean(X)

    Reconstruction error measures how much information is lost.
    Use this to choose the minimum k that preserves enough information.

    Args:
        X        : Feature DataFrame
        k_values : List of n_components values to evaluate
        scale    : Whether to StandardScale

    Returns:
        DataFrame with reconstruction error per k
    """
    if k_values is None:
        max_k    = min(X.shape[0], X.shape[1])
        k_values = list(range(1, min(max_k + 1, 21)))

    X_proc = X.copy()
    scaler = None
    if scale:
        scaler  = StandardScaler()
        X_proc  = scaler.fit_transform(X)
    else:
        X_proc = X.values

    rows = []
    for k in k_values:
        pca   = PCA(n_components=k, svd_solver="full")
        X_red = pca.fit_transform(X_proc)
        X_rec = pca.inverse_transform(X_red)

        mse   = mean_squared_error(X_proc, X_rec)
        evr   = pca.explained_variance_ratio_.sum()

        rows.append({
            "n_components"      : k,
            "Variance Explained": round(evr * 100, 2),
            "MSE (recon error)" : round(mse, 6),
            "RMSE"              : round(np.sqrt(mse), 6),
        })

    df = pd.DataFrame(rows)
    print("Reconstruction Analysis:")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 4. SCREE PLOT DATA (Elbow Method)
# =============================================================================

def scree_analysis(X: pd.DataFrame,
                    scale: bool = True) -> pd.DataFrame:
    """
    Computes eigenvalues and explained variance for all components —
    used to create a scree plot and identify the optimal number of PCs.

    Scree plot: plot eigenvalues vs component number.
    Choose k at the "elbow" — where eigenvalues level off.

    Kaiser's Rule: keep components with eigenvalue > 1.0
    (only when data is standardized)

    Args:
        X     : Feature DataFrame
        scale : Whether to StandardScale

    Returns:
        DataFrame with per-component variance statistics
    """
    X_proc = X.copy()
    if scale:
        X_proc = pd.DataFrame(
            StandardScaler().fit_transform(X), columns=X.columns
        )

    pca  = PCA(svd_solver="full")
    pca.fit(X_proc)

    evr      = pca.explained_variance_ratio_
    ev       = pca.explained_variance_
    cum_evr  = np.cumsum(evr)
    kaiser   = ev > 1.0

    df = pd.DataFrame({
        "Component"         : [f"PC{i+1}" for i in range(len(evr))],
        "Eigenvalue"        : ev.round(4),
        "Variance %"        : (evr * 100).round(2),
        "Cumulative %"      : (cum_evr * 100).round(2),
        "Kaiser (λ>1)"      : kaiser,
    })

    k_kaiser = kaiser.sum()
    k_90     = int(np.searchsorted(cum_evr, 0.90)) + 1
    k_95     = int(np.searchsorted(cum_evr, 0.95)) + 1

    print("Scree Analysis (All Components):")
    print(df.to_string(index=False))
    print(f"\n  Kaiser Rule (λ>1)         : {k_kaiser} components")
    print(f"  90% variance threshold    : {k_90} components")
    print(f"  95% variance threshold    : {k_95} components")

    return df


# =============================================================================
# 🔧 5. LOADINGS ANALYSIS — FEATURE CONTRIBUTIONS
# =============================================================================

def loadings_analysis(pca: PCA,
                        feature_names: list,
                        n_components: int = 3,
                        top_n: int = 5) -> pd.DataFrame:
    """
    Extracts and interprets PCA loadings — how much each original
    feature contributes to each principal component.

    Loadings = elements of eigenvectors (principal axes):
        Large |loading| → feature strongly influences this PC
        Positive loading → feature moves in same direction as PC
        Negative loading → feature moves in opposite direction

    Args:
        pca           : Fitted PCA object
        feature_names : Original feature column names
        n_components  : Number of PCs to analyze (default: 3)
        top_n         : Top N features per PC to display

    Returns:
        DataFrame with loadings per feature per PC
    """
    n_comp_show = min(n_components, pca.n_components_)
    pc_cols     = [f"PC{i+1}" for i in range(n_comp_show)]

    loadings = pd.DataFrame(
        pca.components_[:n_comp_show].T,
        index=feature_names,
        columns=pc_cols
    )

    print(f"[PCA Loadings] Top {top_n} features per component:")
    for pc in pc_cols:
        sorted_loadings = loadings[pc].abs().sort_values(ascending=False)
        top_features    = sorted_loadings.head(top_n).index.tolist()
        print(f"\n  {pc} (top contributors):")
        for feat in top_features:
            val = loadings.loc[feat, pc]
            direction = "↑" if val > 0 else "↓"
            print(f"    {feat:25s} : {val:+.4f}  {direction}")

    return loadings


# =============================================================================
# 🔧 6. PCA PIPELINE (with supervised model)
# =============================================================================

def pca_pipeline(X_train: pd.DataFrame,
                  X_test: pd.DataFrame,
                  y_train: pd.Series,
                  y_test: pd.Series,
                  estimator,
                  n_components=0.95,
                  task: str = "classification") -> dict:
    """
    Builds a StandardScaler → PCA → Estimator pipeline.

    The pipeline ensures no data leakage:
        - Scaler fit only on train
        - PCA fit only on train
        - Estimator fit only on train

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        estimator    : Any sklearn-compatible classifier or regressor
        n_components : PCA components (int, float, or 'mle')
        task         : 'classification' or 'regression'

    Returns:
        Dictionary with pipeline, predictions, and metrics
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_components, random_state=42)),
        ("model",  estimator),
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    pca_step     = pipe.named_steps["pca"]
    n_comp_actual = pca_step.n_components_
    total_var     = pca_step.explained_variance_ratio_.sum()

    if task == "classification":
        from sklearn.metrics import accuracy_score, f1_score
        metrics = {
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "F1"      : round(f1_score(y_test, y_pred,
                                        average="weighted",
                                        zero_division=0), 4),
        }
    else:
        from sklearn.metrics import r2_score
        metrics = {
            "R²"  : round(r2_score(y_test, y_pred), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        }

    print(f"[PCA Pipeline]")
    print(f"  PCA: {X_train.shape[1]}D → {n_comp_actual}D "
          f"({total_var*100:.1f}% variance retained)")
    print(f"  Estimator : {estimator.__class__.__name__}")
    print(f"  Metrics   : {metrics}")

    return {
        "pipeline"     : pipe,
        "y_pred"       : y_pred,
        "metrics"      : metrics,
        "n_components" : n_comp_actual,
        "variance_ret" : total_var,
    }


# =============================================================================
# 🔧 7. KERNEL PCA (Non-Linear)
# =============================================================================

def fit_kernel_pca(X: pd.DataFrame,
                    n_components: int = 2,
                    kernel: str = "rbf",
                    gamma: float = None,
                    degree: int = 3,
                    scale: bool = True,
                    random_state: int = 42) -> dict:
    """
    Fits Kernel PCA for non-linear dimensionality reduction.

    Kernel PCA applies the kernel trick to capture non-linear structure
    in the data that standard (linear) PCA would miss.

    Kernels:
        'rbf'    : Gaussian — most common, good general choice
        'poly'   : Polynomial — captures polynomial interactions
        'cosine' : Cosine similarity — good for text/sparse data
        'linear' : Same as standard PCA

    Args:
        X            : Feature DataFrame
        n_components : Number of components to extract
        kernel       : Kernel type
        gamma        : Kernel coefficient (None = 1/n_features)
        degree       : Degree for polynomial kernel
        scale        : Whether to StandardScale
        random_state : Reproducibility seed

    Returns:
        Dictionary with kpca model, transformed data
    """
    X_proc = X.copy()
    scaler = None

    if scale:
        scaler  = StandardScaler()
        X_proc  = pd.DataFrame(
            scaler.fit_transform(X), columns=X.columns
        )

    kpca = KernelPCA(
        n_components=n_components,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        fit_inverse_transform=True,
        random_state=random_state,
        n_jobs=-1,
    )
    X_kpca = kpca.fit_transform(X_proc)

    pc_cols    = [f"KPC{i+1}" for i in range(n_components)]
    X_kpca_df  = pd.DataFrame(X_kpca, columns=pc_cols)

    print(f"[KernelPCA] kernel={kernel} | n_components={n_components} | "
          f"gamma={gamma}")
    print(f"  Input shape  : {X.shape}")
    print(f"  Output shape : {X_kpca_df.shape}")

    return {
        "kpca"    : kpca,
        "scaler"  : scaler,
        "X_kpca"  : X_kpca_df,
    }


# =============================================================================
# 🔧 8. INCREMENTAL PCA (Large Datasets)
# =============================================================================

def fit_incremental_pca(X: pd.DataFrame,
                          n_components: int = 10,
                          batch_size: int = 200,
                          scale: bool = True) -> dict:
    """
    Fits Incremental PCA for datasets too large to fit in memory.

    Standard PCA requires the full data matrix in memory.
    IncrementalPCA processes data in mini-batches:
        ✅ Much lower memory usage
        ✅ Can handle datasets larger than RAM
        ⚠️ Slightly different results due to batching

    Args:
        X            : Feature DataFrame
        n_components : Number of PCs to extract
        batch_size   : Samples per mini-batch (default: 200)
        scale        : Whether to StandardScale

    Returns:
        Dictionary with ipca model, transformed data, and stats
    """
    X_proc = X.copy()
    scaler = None

    if scale:
        scaler  = StandardScaler()
        X_proc  = pd.DataFrame(
            scaler.fit_transform(X), columns=X.columns
        )

    ipca  = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    X_red = ipca.fit_transform(X_proc)

    evr     = ipca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)
    pc_cols = [f"PC{i+1}" for i in range(n_components)]
    X_df    = pd.DataFrame(X_red, columns=pc_cols)

    print(f"[IncrementalPCA] n_components={n_components} | "
          f"batch_size={batch_size}")
    print(f"  Input shape    : {X.shape}")
    print(f"  Output shape   : {X_df.shape}")
    print(f"  Variance (top5): "
          f"{[round(v*100,2) for v in evr[:5]]}%")
    print(f"  Cumulative     : {round(cum_evr[-1]*100,2)}% total")

    return {
        "ipca"         : ipca,
        "scaler"       : scaler,
        "X_pca"        : X_df,
        "explained_var": evr,
        "cumulative_var": cum_evr,
    }


# =============================================================================
# 🔧 9. PCA FOR VISUALIZATION (2D / 3D)
# =============================================================================

def pca_for_visualization(X: pd.DataFrame,
                            y: pd.Series = None,
                            n_components: int = 2,
                            scale: bool = True) -> pd.DataFrame:
    """
    Reduces data to 2D or 3D for visualization purposes.

    Args:
        X            : Feature DataFrame
        y            : Optional label Series to attach for coloring
        n_components : 2 (2D plot) or 3 (3D plot)
        scale        : Whether to StandardScale

    Returns:
        DataFrame with PC1, PC2 (and optionally PC3 and label columns)
    """
    X_proc = X.copy()
    if scale:
        X_proc = pd.DataFrame(
            StandardScaler().fit_transform(X), columns=X.columns
        )

    pca   = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_proc)
    evr   = pca.explained_variance_ratio_

    pc_cols  = [f"PC{i+1}" for i in range(n_components)]
    result_df = pd.DataFrame(X_pca, columns=pc_cols)

    if y is not None:
        result_df["Label"] = y.values if hasattr(y, "values") else y

    print(f"[PCA Visualization] {X.shape[1]}D → {n_components}D")
    for i, (col, var) in enumerate(zip(pc_cols, evr)):
        print(f"  {col}: {var*100:.2f}% variance explained")
    print(f"  Total: {evr.sum()*100:.2f}% variance retained")

    return result_df


# =============================================================================
# 🔧 10. COMPARE ORIGINAL vs PCA PIPELINE PERFORMANCE
# =============================================================================

def compare_with_without_pca(X_train: pd.DataFrame,
                               X_test: pd.DataFrame,
                               y_train: pd.Series,
                               y_test: pd.Series,
                               estimator,
                               n_components_list: list = None,
                               task: str = "classification") -> pd.DataFrame:
    """
    Compares model performance with and without PCA preprocessing,
    across multiple n_components values.

    Useful for finding the minimum dimensionality that preserves
    predictive performance.

    Args:
        X_train, X_test, y_train, y_test : Train/test split
        estimator          : sklearn classifier or regressor
        n_components_list  : List of n_components to evaluate
        task               : 'classification' or 'regression'

    Returns:
        DataFrame comparing performance across dimensionalities
    """
    import copy

    if n_components_list is None:
        max_k = min(X_train.shape[1], 20)
        n_components_list = list(range(1, max_k + 1))

    if task == "classification":
        from sklearn.metrics import accuracy_score, f1_score
        def score(yt, yp):
            return {
                "Accuracy": round(accuracy_score(yt, yp), 4),
                "F1"      : round(f1_score(yt, yp, average="weighted",
                                            zero_division=0), 4),
            }
    else:
        from sklearn.metrics import r2_score
        def score(yt, yp):
            return {
                "R²"  : round(r2_score(yt, yp), 4),
                "RMSE": round(np.sqrt(mean_squared_error(yt, yp)), 4),
            }

    rows = []

    # Baseline: no PCA
    scaler_base = StandardScaler()
    X_tr_sc     = scaler_base.fit_transform(X_train)
    X_te_sc     = scaler_base.transform(X_test)
    est_base    = copy.deepcopy(estimator)
    est_base.fit(X_tr_sc, y_train)
    base_metrics = score(y_test, est_base.predict(X_te_sc))
    base_row     = {
        "n_components"      : X_train.shape[1],
        "Variance Retained" : 100.0,
        "Method"            : "No PCA (baseline)",
    }
    base_row.update(base_metrics)
    rows.append(base_row)

    # With PCA
    for k in n_components_list:
        if k >= X_train.shape[1]:
            continue
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=k, random_state=42)),
            ("model",  copy.deepcopy(estimator)),
        ])
        pipe.fit(X_train, y_train)
        yp   = pipe.predict(X_test)
        var  = pipe.named_steps["pca"].explained_variance_ratio_.sum()
        m    = score(y_test, yp)
        row  = {
            "n_components"      : k,
            "Variance Retained" : round(var * 100, 2),
            "Method"            : f"PCA(k={k})",
        }
        row.update(m)
        rows.append(row)

    df = pd.DataFrame(rows)
    print("Performance Comparison — With vs Without PCA:")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 HELPERS
# =============================================================================

def variance_threshold_k(pca_result: dict,
                           threshold: float = 0.95) -> int:
    """Returns the minimum k to retain `threshold` fraction of variance."""
    cum_var = pca_result["cumulative_var"]
    for k, cv in enumerate(cum_var):
        if cv >= threshold:
            print(f"  {threshold*100:.0f}% variance → k = {k+1} components "
                  f"(cumulative: {cv*100:.2f}%)")
            return k + 1
    print(f"  Need all {len(cum_var)} components for {threshold*100:.0f}% variance")
    return len(cum_var)


# =============================================================================
# 🚀 MAIN — Demo with Synthetic Dataset
# =============================================================================

if __name__ == "__main__":

    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    np.random.seed(42)

    # ── High-dimensional synthetic dataset ────────────────────────────────
    X_raw, y_raw = make_classification(
        n_samples=800, n_features=30, n_informative=10,
        n_redundant=10, n_repeated=5, n_classes=2,
        random_state=42
    )
    X = pd.DataFrame(X_raw, columns=[f"Feature_{i+1:02d}" for i in range(30)])
    y = pd.Series(y_raw, name="Target")

    print("=" * 65)
    print("📊 Dataset Info — High-Dimensional (30 features)")
    print("=" * 65)
    print(f"Shape: {X.shape} | Classes: {dict(y.value_counts().sort_index())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── 1. Fit PCA (95% variance) ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("1️⃣  Fit PCA (retain 95% variance)")
    print("=" * 65)
    result = fit_pca(X_train, n_components=0.95)
    k_95   = variance_threshold_k(result, threshold=0.95)

    # ── 2. Scree Analysis ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("2️⃣  Scree Analysis — All Components")
    print("=" * 65)
    scree_df = scree_analysis(X_train)

    # ── 3. Loadings Analysis ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("3️⃣  Loadings Analysis — Top Feature Contributions")
    print("=" * 65)
    loadings_df = loadings_analysis(
        result["pca"],
        feature_names=X_train.columns.tolist(),
        n_components=3, top_n=5
    )

    # ── 4. Reconstruction Analysis ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("4️⃣  Reconstruction Error vs n_components")
    print("=" * 65)
    recon_df = reconstruction_analysis(X_train, k_values=list(range(1, 21)))

    # ── 5. Transform Test Data ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("5️⃣  Transform Test Data (no leakage)")
    print("=" * 65)
    X_test_pca = transform_pca(result["pca"], result["scaler"], X_test)
    print(f"  Test shape : {X_test.shape} → {X_test_pca.shape}")

    # ── 6. PCA for Visualization (2D) ─────────────────────────────────────
    print("\n" + "=" * 65)
    print("6️⃣  PCA for 2D Visualization")
    print("=" * 65)
    viz_df = pca_for_visualization(X_train, y=y_train, n_components=2)
    print(viz_df.head())

    # ── 7. PCA Pipeline + Classifier ─────────────────────────────────────
    print("\n" + "=" * 65)
    print("7️⃣  PCA Pipeline with Logistic Regression")
    print("=" * 65)
    pipe_result = pca_pipeline(
        X_train, X_test, y_train, y_test,
        estimator=LogisticRegression(max_iter=1000, random_state=42),
        n_components=0.95,
        task="classification"
    )

    # ── 8. Compare With vs Without PCA ────────────────────────────────────
    print("\n" + "=" * 65)
    print("8️⃣  Compare Performance With vs Without PCA")
    print("=" * 65)
    comp_df = compare_with_without_pca(
        X_train, X_test, y_train, y_test,
        estimator=LogisticRegression(max_iter=1000, random_state=42),
        n_components_list=[2, 5, 10, 15, 20, 25],
        task="classification"
    )

    # ── 9. Kernel PCA ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("9️⃣  Kernel PCA (RBF — Non-Linear)")
    print("=" * 65)
    kpca_result = fit_kernel_pca(X_train, n_components=2, kernel="rbf")
    print(kpca_result["X_kpca"].head())

    # ── 10. Incremental PCA ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("🔟  Incremental PCA (Large Dataset Mode)")
    print("=" * 65)
    ipca_result = fit_incremental_pca(
        X_train, n_components=10, batch_size=100
    )

    print("\n✅ All PCA techniques demonstrated successfully!")
