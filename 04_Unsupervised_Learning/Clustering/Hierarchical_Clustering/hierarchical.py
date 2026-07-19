# =============================================================================
# 📦 Hierarchical Clustering — Reusable ML Script
# =============================================================================
# Author   : Hamna Munir
# Topic    : 04_Unsupervised_Learning / Hierarchical_Clustering
# File     : hierarchical.py
# =============================================================================

# -----------------------------------------------------------------------------
# 📚 Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score
)
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import (
    linkage, dendrogram, fcluster,
    cophenet, inconsistent
)
from scipy.spatial.distance import pdist

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 🔧 1. TRAIN AGGLOMERATIVE CLUSTERING (sklearn)
# =============================================================================

def train_agglomerative(X: pd.DataFrame,
                         n_clusters: int = 3,
                         linkage_method: str = "ward",
                         metric: str = "euclidean",
                         scale: bool = True) -> dict:
    """
    Trains Agglomerative (bottom-up) Hierarchical Clustering.

    Algorithm:
        1. Each point starts as its own cluster
        2. Iteratively merge the two closest clusters
        3. Distance between clusters defined by linkage criterion
        4. Stop when n_clusters remain

    Linkage options:
        'ward'     : Minimizes within-cluster variance (default, best general)
        'complete' : Max distance between points in two clusters (compact)
        'average'  : Mean distance between all cross-cluster pairs (robust)
        'single'   : Min distance — can find elongated shapes but chains

    ⚠️ Ward linkage requires Euclidean metric.
       For other metrics (manhattan, cosine), use average or complete.

    Args:
        X              : Feature DataFrame
        n_clusters     : Number of clusters to extract
        linkage_method : 'ward', 'complete', 'average', 'single'
        metric         : Distance metric — 'euclidean', 'manhattan', 'cosine'
        scale          : Whether to StandardScale features (default: True)

    Returns:
        Dictionary with model, labels, and evaluation metrics
    """
    X_proc = X.copy()
    scaler = None

    if scale:
        scaler  = StandardScaler()
        X_proc  = pd.DataFrame(
            scaler.fit_transform(X), columns=X.columns
        )

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method,
        metric=metric,
    )
    labels = model.fit_predict(X_proc)

    sil_score = silhouette_score(X_proc, labels) if n_clusters > 1 else None
    ch_score  = calinski_harabasz_score(X_proc, labels)
    db_score  = davies_bouldin_score(X_proc, labels)
    sizes     = pd.Series(labels).value_counts().sort_index()

    print(f"[AgglomerativeClustering] n_clusters={n_clusters} | "
          f"linkage={linkage_method} | metric={metric}")
    print(f"  Silhouette Score  : {sil_score:.4f}" if sil_score else "")
    print(f"  Calinski-Harabasz : {ch_score:.4f}")
    print(f"  Davies-Bouldin    : {db_score:.4f}")
    print(f"  Cluster sizes     : {sizes.to_dict()}")

    return {
        "model"         : model,
        "scaler"        : scaler,
        "labels"        : labels,
        "X_scaled"      : X_proc,
        "silhouette"    : sil_score,
        "calinski"      : ch_score,
        "davies_bouldin": db_score,
        "cluster_sizes" : sizes,
    }


# =============================================================================
# 🔧 2. BUILD LINKAGE MATRIX (scipy — for dendrogram)
# =============================================================================

def build_linkage_matrix(X: pd.DataFrame,
                           method: str = "ward",
                           metric: str = "euclidean",
                           scale: bool = True) -> tuple:
    """
    Computes the scipy linkage matrix — required for plotting dendrograms
    and cutting the tree at arbitrary heights.

    The linkage matrix Z encodes the full merge history:
        Z[i] = [cluster_a, cluster_b, distance, count]
        Each row = one merge step

    Args:
        X      : Feature DataFrame
        method : Linkage method — 'ward', 'complete', 'average', 'single'
        metric : Distance metric
        scale  : Whether to StandardScale

    Returns:
        Tuple of (Z, X_scaled, scaler)
        Z = linkage matrix (scipy format)
    """
    X_proc = X.copy()
    scaler = None

    if scale:
        scaler  = StandardScaler()
        X_proc  = pd.DataFrame(
            scaler.fit_transform(X), columns=X.columns
        )

    Z = linkage(X_proc, method=method, metric=metric)

    # Cophenetic correlation — goodness of dendrogram fit
    c, _ = cophenet(Z, pdist(X_proc, metric=metric))

    print(f"[Linkage Matrix] method={method} | metric={metric}")
    print(f"  Shape              : {Z.shape}  ({len(X)} merges)")
    print(f"  Cophenetic corr    : {c:.4f}  "
          f"(1.0 = perfect dendrogram fit)")
    print(f"  Height range       : [{Z[:,2].min():.3f}, {Z[:,2].max():.3f}]")

    return Z, X_proc.values if isinstance(X_proc, pd.DataFrame) else X_proc, scaler


# =============================================================================
# 🔧 3. CUT DENDROGRAM — EXTRACT LABELS
# =============================================================================

def cut_dendrogram(Z: np.ndarray,
                    criterion: str = "maxclust",
                    t: float = 3.0) -> np.ndarray:
    """
    Cuts the dendrogram at a specified level to extract cluster labels.

    Cutting criteria:
        'maxclust'     : Cut so exactly t clusters result
        'distance'     : Cut at height t (merge distance threshold)
        'inconsistent' : Cut based on inconsistency coefficient

    Args:
        Z         : Linkage matrix from build_linkage_matrix()
        criterion : How to cut — 'maxclust', 'distance', 'inconsistent'
        t         : Threshold value:
                    maxclust   → integer number of clusters
                    distance   → float height to cut at
                    inconsistent → inconsistency threshold

    Returns:
        Array of cluster labels (1-indexed from scipy)
    """
    labels = fcluster(Z, t=t, criterion=criterion)
    labels = labels - 1  # Convert 1-indexed → 0-indexed for consistency

    unique, counts = np.unique(labels, return_counts=True)
    print(f"[Cut Dendrogram] criterion={criterion} | t={t}")
    print(f"  Clusters found: {len(unique)}")
    print(f"  Cluster sizes : {dict(zip(unique, counts))}")

    return labels


# =============================================================================
# 🔧 4. LINKAGE METHOD COMPARISON
# =============================================================================

def compare_linkage_methods(X: pd.DataFrame,
                              n_clusters: int = 3,
                              scale: bool = True) -> pd.DataFrame:
    """
    Compares all four linkage methods on quality metrics.

    The best linkage depends on the data geometry:
        ward     → compact, spherical clusters (most datasets)
        average  → robust general-purpose choice
        complete → compact, avoids chaining
        single   → elongated, non-convex shapes

    Args:
        X          : Feature DataFrame
        n_clusters : Number of clusters (same for all methods)
        scale      : Whether to StandardScale

    Returns:
        DataFrame with metrics per linkage method
    """
    X_proc = X.copy()
    if scale:
        X_proc = pd.DataFrame(
            StandardScaler().fit_transform(X), columns=X.columns
        )

    methods = ["ward", "complete", "average", "single"]
    rows = []

    for method in methods:
        metric = "euclidean"
        model  = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=method,
            metric=metric,
        )
        labels = model.fit_predict(X_proc)
        sil    = silhouette_score(X_proc, labels)
        ch     = calinski_harabasz_score(X_proc, labels)
        db     = davies_bouldin_score(X_proc, labels)
        sizes  = pd.Series(labels).value_counts().sort_index().to_dict()

        rows.append({
            "Linkage"          : method,
            "Silhouette"       : round(sil, 4),
            "Calinski-Harabasz": round(ch, 4),
            "Davies-Bouldin"   : round(db, 4),
            "Cluster Sizes"    : str(sizes),
        })

    df = pd.DataFrame(rows)
    best = df.loc[df["Silhouette"].idxmax(), "Linkage"]
    print(f"Linkage Method Comparison (n_clusters={n_clusters}):")
    print(df.to_string(index=False))
    print(f"\n  Best linkage by Silhouette: {best}")
    return df


# =============================================================================
# 🔧 5. METRIC COMPARISON
# =============================================================================

def compare_distance_metrics(X: pd.DataFrame,
                               n_clusters: int = 3,
                               linkage_method: str = "average",
                               scale: bool = True) -> pd.DataFrame:
    """
    Compares different distance metrics with a fixed linkage method.

    Note: Ward linkage only supports Euclidean metric.
          Use 'average' or 'complete' when testing other metrics.

    Args:
        X              : Feature DataFrame
        n_clusters     : Number of clusters
        linkage_method : Linkage method (avoid 'ward' here)
        scale          : Whether to StandardScale

    Returns:
        DataFrame with metrics per distance metric
    """
    X_proc = X.copy()
    if scale:
        X_proc = pd.DataFrame(
            StandardScaler().fit_transform(X), columns=X.columns
        )

    metrics_list = ["euclidean", "manhattan", "cosine"]
    rows = []

    for metric in metrics_list:
        model  = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method,
            metric=metric,
        )
        labels = model.fit_predict(X_proc)
        sil    = silhouette_score(X_proc, labels,
                                   metric=metric)
        ch     = calinski_harabasz_score(X_proc, labels)
        db     = davies_bouldin_score(X_proc, labels)

        rows.append({
            "Metric"           : metric,
            "Silhouette"       : round(sil, 4),
            "Calinski-Harabasz": round(ch, 4),
            "Davies-Bouldin"   : round(db, 4),
        })

    df = pd.DataFrame(rows)
    print(f"Distance Metric Comparison (linkage={linkage_method}, "
          f"n_clusters={n_clusters}):")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 6. SILHOUETTE ANALYSIS ACROSS K VALUES
# =============================================================================

def silhouette_analysis(X: pd.DataFrame,
                          k_range: range = None,
                          linkage_method: str = "ward",
                          scale: bool = True) -> pd.DataFrame:
    """
    Evaluates clustering quality across a range of K values
    by cutting the dendrogram at different levels.

    Args:
        X              : Feature DataFrame
        k_range        : Range of K to evaluate (default: 2–15)
        linkage_method : Linkage method for all evaluations
        scale          : Whether to StandardScale

    Returns:
        DataFrame with K and quality metrics
    """
    if k_range is None:
        k_range = range(2, 16)

    X_proc = X.copy()
    if scale:
        X_proc = pd.DataFrame(
            StandardScaler().fit_transform(X), columns=X.columns
        )

    rows = []
    for k in k_range:
        model  = AgglomerativeClustering(
            n_clusters=k, linkage=linkage_method
        )
        labels = model.fit_predict(X_proc)
        sil    = silhouette_score(X_proc, labels)
        ch     = calinski_harabasz_score(X_proc, labels)
        db     = davies_bouldin_score(X_proc, labels)

        rows.append({
            "K"                : k,
            "Silhouette"       : round(sil, 4),
            "Calinski-Harabasz": round(ch, 4),
            "Davies-Bouldin"   : round(db, 4),
        })

    df = pd.DataFrame(rows)
    best_k = df.loc[df["Silhouette"].idxmax(), "K"]
    print(f"Silhouette Analysis (linkage={linkage_method}):")
    print(df.to_string(index=False))
    print(f"\n  Optimal K (silhouette): {best_k}")
    return df


# =============================================================================
# 🔧 7. CLUSTER PROFILE
# =============================================================================

def cluster_profile(X: pd.DataFrame,
                     labels: np.ndarray,
                     top_n: int = 5) -> pd.DataFrame:
    """
    Builds a statistical profile of each cluster —
    shows mean feature values per cluster to understand
    what distinguishes each group.

    Args:
        X      : Original (unscaled) feature DataFrame
        labels : Cluster assignment array
        top_n  : Top N most discriminative features to highlight

    Returns:
        DataFrame with per-cluster feature means
    """
    X_prof = X.copy()
    X_prof["Cluster"] = labels

    means = X_prof.groupby("Cluster").mean().round(4)
    sizes = X_prof.groupby("Cluster").size().rename("Count")
    profile = pd.concat([sizes, means], axis=1)

    print(f"[Cluster Profile] {labels.max()+1} clusters")
    print(profile.to_string())

    # Most discriminative = highest variance across cluster means
    cluster_means   = X_prof.groupby("Cluster").mean()
    feat_variance   = cluster_means.var().sort_values(ascending=False)
    top_features    = feat_variance.head(top_n)
    print(f"\n  Top {top_n} most discriminative features:")
    print(top_features.round(4).to_string())

    return profile


# =============================================================================
# 🔧 8. EVALUATE AGAINST GROUND TRUTH LABELS
# =============================================================================

def evaluate_with_labels(labels_pred: np.ndarray,
                           labels_true: np.ndarray,
                           model_name: str = "Hierarchical") -> pd.DataFrame:
    """
    Evaluates clustering quality using ground truth labels.

    Args:
        labels_pred : Predicted cluster labels
        labels_true : True class labels
        model_name  : Name for display

    Returns:
        DataFrame with evaluation metrics
    """
    metrics = {
        "Model"       : model_name,
        "ARI"         : round(adjusted_rand_score(labels_true, labels_pred), 4),
        "NMI"         : round(normalized_mutual_info_score(
                                  labels_true, labels_pred), 4),
        "Homogeneity" : round(homogeneity_score(labels_true, labels_pred), 4),
        "Completeness": round(completeness_score(labels_true, labels_pred), 4),
        "V-Measure"   : round(v_measure_score(labels_true, labels_pred), 4),
    }

    report = pd.DataFrame([metrics])
    print(f"\n📊 Cluster Evaluation (vs Ground Truth) — {model_name}")
    print(report.to_string(index=False))
    return report


# =============================================================================
# 🔧 9. HIERARCHICAL + PCA VISUALIZATION
# =============================================================================

def hierarchical_with_pca(X: pd.DataFrame,
                            n_clusters: int = 3,
                            linkage_method: str = "ward",
                            n_components: int = 2,
                            scale: bool = True) -> dict:
    """
    Runs Agglomerative Clustering and projects to 2D via PCA
    for cluster visualization.

    Args:
        X              : Feature DataFrame
        n_clusters     : Number of clusters
        linkage_method : Linkage method
        n_components   : PCA components for projection
        scale          : Whether to StandardScale

    Returns:
        Dictionary with labels, PCA-reduced DataFrame, explained variance
    """
    X_proc = X.copy()
    scaler = None

    if scale:
        scaler  = StandardScaler()
        X_proc  = pd.DataFrame(
            scaler.fit_transform(X), columns=X.columns
        )

    model  = AgglomerativeClustering(
        n_clusters=n_clusters, linkage=linkage_method
    )
    labels = model.fit_predict(X_proc)

    pca     = PCA(n_components=n_components, random_state=42)
    X_pca   = pca.fit_transform(X_proc)
    exp_var = pca.explained_variance_ratio_

    pca_cols = [f"PC{i+1}" for i in range(n_components)]
    X_pca_df = pd.DataFrame(X_pca, columns=pca_cols)
    X_pca_df["Cluster"] = labels

    sil = silhouette_score(X_proc, labels)

    print(f"[Hierarchical + PCA] n_clusters={n_clusters} | "
          f"linkage={linkage_method}")
    print(f"  Silhouette   : {sil:.4f}")
    print(f"  Explained var: "
          f"{[round(v*100,2) for v in exp_var]}% "
          f"(total: {sum(exp_var)*100:.1f}%)")

    return {
        "model"       : model,
        "scaler"      : scaler,
        "pca"         : pca,
        "labels"      : labels,
        "X_pca"       : X_pca_df,
        "explained_var": exp_var,
        "silhouette"  : sil,
    }


# =============================================================================
# 🔧 10. COPHENETIC CORRELATION — DENDROGRAM QUALITY
# =============================================================================

def cophenetic_correlation(X: pd.DataFrame,
                             methods: list = None,
                             scale: bool = True) -> pd.DataFrame:
    """
    Computes the cophenetic correlation coefficient for each linkage method.

    Cophenetic correlation measures how faithfully the dendrogram
    preserves the pairwise distances of the original data:
        cc = 1.0 → perfect preservation
        cc > 0.8 → good dendrogram
        cc < 0.7 → poor dendrogram

    Args:
        X       : Feature DataFrame
        methods : List of linkage methods to evaluate
        scale   : Whether to StandardScale

    Returns:
        DataFrame with cophenetic correlation per method
    """
    if methods is None:
        methods = ["ward", "complete", "average", "single"]

    X_proc = X.copy()
    if scale:
        X_proc = pd.DataFrame(
            StandardScaler().fit_transform(X), columns=X.columns
        )

    dist_condensed = pdist(X_proc, metric="euclidean")
    rows = []

    for method in methods:
        Z    = linkage(X_proc, method=method, metric="euclidean")
        cc, _ = cophenet(Z, dist_condensed)
        rows.append({
            "Linkage Method"       : method,
            "Cophenetic Correlation": round(cc, 4),
            "Quality"              : "✅ Good" if cc > 0.8 else
                                     "⚠️ Moderate" if cc > 0.7 else
                                     "❌ Poor",
        })

    df = pd.DataFrame(rows).sort_values(
        "Cophenetic Correlation", ascending=False
    ).reset_index(drop=True)

    print("Cophenetic Correlation per Linkage Method:")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 11. FULL PIPELINE — AUTO SELECT K + FIT
# =============================================================================

def auto_hierarchical(X: pd.DataFrame,
                       k_range: range = None,
                       linkage_method: str = "ward",
                       scale: bool = True,
                       random_state: int = 42) -> dict:
    """
    Full auto-pipeline:
        1. StandardScale
        2. Silhouette analysis to find best K
        3. Fit AgglomerativeClustering with best K
        4. Return labels + all metrics

    Args:
        X              : Feature DataFrame
        k_range        : K values to search (default: 2–10)
        linkage_method : Linkage method
        scale          : Whether to StandardScale
        random_state   : For reproducibility

    Returns:
        Dictionary with best K, labels, and metrics
    """
    if k_range is None:
        k_range = range(2, 11)

    X_proc = X.copy()
    scaler = None

    if scale:
        scaler  = StandardScaler()
        X_proc  = pd.DataFrame(
            scaler.fit_transform(X), columns=X.columns
        )

    # Step 1: find best K
    best_k   = 2
    best_sil = -1
    for k in k_range:
        model  = AgglomerativeClustering(
            n_clusters=k, linkage=linkage_method
        )
        labels = model.fit_predict(X_proc)
        sil    = silhouette_score(X_proc, labels)
        if sil > best_sil:
            best_sil = sil
            best_k   = k

    # Step 2: fit with best K
    model  = AgglomerativeClustering(
        n_clusters=best_k, linkage=linkage_method
    )
    labels = model.fit_predict(X_proc)

    sil = silhouette_score(X_proc, labels)
    ch  = calinski_harabasz_score(X_proc, labels)
    db  = davies_bouldin_score(X_proc, labels)

    print(f"[Auto Hierarchical] best_K={best_k} | "
          f"linkage={linkage_method}")
    print(f"  Silhouette   : {sil:.4f}")
    print(f"  Calinski     : {ch:.4f}")
    print(f"  Davies-Bouldin: {db:.4f}")
    print(f"  Cluster sizes: "
          f"{pd.Series(labels).value_counts().sort_index().to_dict()}")

    return {
        "model"         : model,
        "scaler"        : scaler,
        "labels"        : labels,
        "X_scaled"      : X_proc,
        "best_k"        : best_k,
        "silhouette"    : sil,
        "calinski"      : ch,
        "davies_bouldin": db,
    }


# =============================================================================
# 🔧 HELPERS
# =============================================================================

def summarize_clusters(X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Quick cluster size and mean summary."""
    X_copy = X.copy()
    X_copy["Cluster"] = labels
    summary = X_copy.groupby("Cluster").agg(
        Count=("Cluster", "size"),
        **{f"{col}_mean": (col, "mean") for col in X.columns}
    ).round(4)
    print("[Cluster Summary]")
    print(summary.to_string())
    return summary


# =============================================================================
# 🚀 MAIN — Demo with Synthetic Dataset
# =============================================================================

if __name__ == "__main__":

    from sklearn.datasets import make_blobs

    np.random.seed(42)

    # ── Synthetic clustered dataset ───────────────────────────────────────
    X_raw, y_true = make_blobs(
        n_samples=300,
        n_features=5,
        centers=4,
        cluster_std=1.1,
        random_state=42
    )
    X = pd.DataFrame(X_raw, columns=[f"Feature_{i+1}" for i in range(5)])

    print("=" * 65)
    print("📊 Dataset Info — Synthetic Blobs")
    print("=" * 65)
    print(f"Shape      : {X.shape}")
    print(f"True groups: 4")

    # ── 1. Agglomerative Clustering ───────────────────────────────────────
    print("\n" + "=" * 65)
    print("1️⃣  Agglomerative Clustering (ward, K=4)")
    print("=" * 65)
    result = train_agglomerative(X, n_clusters=4, linkage_method="ward")

    # ── 2. Build Linkage Matrix ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("2️⃣  Build Linkage Matrix (scipy)")
    print("=" * 65)
    Z, X_sc, scaler = build_linkage_matrix(X, method="ward")

    # ── 3. Cut Dendrogram ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("3️⃣  Cut Dendrogram (maxclust=4)")
    print("=" * 65)
    cut_labels = cut_dendrogram(Z, criterion="maxclust", t=4)

    # ── 4. Linkage Comparison ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("4️⃣  Linkage Method Comparison")
    print("=" * 65)
    link_df = compare_linkage_methods(X, n_clusters=4)

    # ── 5. Metric Comparison ──────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("5️⃣  Distance Metric Comparison (average linkage)")
    print("=" * 65)
    metric_df = compare_distance_metrics(X, n_clusters=4,
                                          linkage_method="average")

    # ── 6. Silhouette Analysis ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("6️⃣  Silhouette Analysis — Find Optimal K")
    print("=" * 65)
    sil_df = silhouette_analysis(X, k_range=range(2, 10))

    # ── 7. Cophenetic Correlation ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("7️⃣  Cophenetic Correlation (Dendrogram Quality)")
    print("=" * 65)
    cc_df = cophenetic_correlation(X)

    # ── 8. Cluster Profile ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("8️⃣  Cluster Profile")
    print("=" * 65)
    profile = cluster_profile(X, result["labels"])

    # ── 9. Evaluate vs Ground Truth ───────────────────────────────────────
    print("\n" + "=" * 65)
    print("9️⃣  Evaluate vs Ground Truth Labels")
    print("=" * 65)
    eval_df = evaluate_with_labels(result["labels"], y_true)

    # ── 10. Hierarchical + PCA ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("🔟  Hierarchical + PCA (2D projection)")
    print("=" * 65)
    pca_result = hierarchical_with_pca(X, n_clusters=4)
    print(pca_result["X_pca"].head())

    # ── 11. Auto Pipeline ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("1️⃣1️⃣  Auto Hierarchical (auto K selection)")
    print("=" * 65)
    auto_result = auto_hierarchical(X, k_range=range(2, 8))

    print("\n✅ All Hierarchical Clustering techniques demonstrated successfully!")
