# =============================================================================
# 📦 K-Means Clustering — Reusable ML Script
# =============================================================================
# Author   : Hamna Munir
# Topic    : 04_Unsupervised_Learning / KMeans
# File     : kmeans.py
# =============================================================================

# -----------------------------------------------------------------------------
# 📚 Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score
)
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 🔧 1. TRAIN K-MEANS CLUSTERING
# =============================================================================

def train_kmeans(X: pd.DataFrame,
                  n_clusters: int = 3,
                  init: str = "k-means++",
                  n_init: int = 10,
                  max_iter: int = 300,
                  scale: bool = True,
                  random_state: int = 42) -> dict:
    """
    Trains a K-Means clustering model.

    Algorithm (Lloyd's):
        1. Initialize K centroids (k-means++ by default)
        2. Assign each point to nearest centroid
        3. Update centroids = mean of assigned points
        4. Repeat 2–3 until convergence

    ⚠️ Always scale features before K-Means — it is distance-based
        and dominated by large-range features without scaling.

    Args:
        X            : Feature DataFrame (no labels)
        n_clusters   : Number of clusters K
        init         : Centroid initialization — 'k-means++' or 'random'
        n_init       : Number of random restarts (best result kept)
        max_iter     : Max iterations per run
        scale        : Whether to StandardScale features (default: True)
        random_state : Reproducibility seed

    Returns:
        Dictionary with model, labels, centroids, inertia, and metrics
    """
    X_proc = X.copy()
    scaler = None

    if scale:
        scaler  = StandardScaler()
        X_proc  = pd.DataFrame(
            scaler.fit_transform(X), columns=X.columns
        )

    model = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(X_proc)

    labels    = model.labels_
    inertia   = model.inertia_
    centroids = pd.DataFrame(
        model.cluster_centers_, columns=X.columns
    )

    # Evaluation metrics (no labels needed)
    sil_score = silhouette_score(X_proc, labels) if n_clusters > 1 else None
    ch_score  = calinski_harabasz_score(X_proc, labels)
    db_score  = davies_bouldin_score(X_proc, labels)

    cluster_sizes = pd.Series(labels).value_counts().sort_index()

    print(f"[KMeans] n_clusters={n_clusters} | init={init} | n_init={n_init}")
    print(f"  Inertia (WCSS)      : {inertia:.4f}")
    print(f"  Silhouette Score    : {sil_score:.4f}" if sil_score else "")
    print(f"  Calinski-Harabasz   : {ch_score:.4f}")
    print(f"  Davies-Bouldin      : {db_score:.4f}")
    print(f"  Cluster sizes       : {cluster_sizes.to_dict()}")

    return {
        "model"        : model,
        "scaler"       : scaler,
        "labels"       : labels,
        "X_scaled"     : X_proc,
        "centroids"    : centroids,
        "inertia"      : inertia,
        "silhouette"   : sil_score,
        "calinski"     : ch_score,
        "davies_bouldin": db_score,
        "cluster_sizes": cluster_sizes,
        "n_iter"       : model.n_iter_,
    }


# =============================================================================
# 🔧 2. ELBOW METHOD — FIND OPTIMAL K
# =============================================================================

def elbow_method(X: pd.DataFrame,
                  k_range: range = None,
                  scale: bool = True,
                  random_state: int = 42) -> pd.DataFrame:
    """
    Runs K-Means for a range of K values and collects inertia —
    used to identify the elbow point (optimal K).

    The elbow is where inertia drops sharply then levels off —
    adding more clusters gives diminishing returns.

    Args:
        X            : Feature DataFrame
        k_range      : Range of K values (default: 1–15)
        scale        : Whether to StandardScale features
        random_state : Reproducibility seed

    Returns:
        DataFrame with K, inertia, and delta-inertia per K
    """
    if k_range is None:
        k_range = range(1, 16)

    X_proc = X.copy()
    if scale:
        X_proc = pd.DataFrame(
            StandardScaler().fit_transform(X), columns=X.columns
        )

    rows = []
    for k in k_range:
        model = KMeans(
            n_clusters=k, init="k-means++", n_init=10,
            random_state=random_state
        )
        model.fit(X_proc)
        rows.append({"K": k, "Inertia": round(model.inertia_, 4)})

    df = pd.DataFrame(rows)
    df["Delta Inertia"] = df["Inertia"].diff().abs().round(4)
    df["Delta %"] = (
        df["Inertia"].pct_change().abs() * 100
    ).round(2)

    print("Elbow Method — Inertia vs K:")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 3. SILHOUETTE ANALYSIS — OPTIMAL K
# =============================================================================

def silhouette_analysis(X: pd.DataFrame,
                          k_range: range = None,
                          scale: bool = True,
                          random_state: int = 42) -> pd.DataFrame:
    """
    Computes silhouette scores for a range of K values —
    the K with the HIGHEST silhouette score is optimal.

    Silhouette score per point:
        s(i) = (b(i) − a(i)) / max(a(i), b(i))
        a(i) = mean intra-cluster distance
        b(i) = mean distance to nearest other cluster
        Range: [−1, 1] — higher is better

    Args:
        X            : Feature DataFrame
        k_range      : Range of K to evaluate (default: 2–15)
        scale        : Whether to StandardScale
        random_state : Reproducibility seed

    Returns:
        DataFrame with K and silhouette score
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
        model = KMeans(
            n_clusters=k, init="k-means++", n_init=10,
            random_state=random_state
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
    best_k_sil = df.loc[df["Silhouette"].idxmax(), "K"]
    best_k_ch  = df.loc[df["Calinski-Harabasz"].idxmax(), "K"]
    best_k_db  = df.loc[df["Davies-Bouldin"].idxmin(), "K"]

    print("Silhouette Analysis:")
    print(df.to_string(index=False))
    print(f"\n  Best K by Silhouette       : {best_k_sil}")
    print(f"  Best K by Calinski-Harabasz: {best_k_ch}")
    print(f"  Best K by Davies-Bouldin   : {best_k_db}")
    return df


# =============================================================================
# 🔧 4. SILHOUETTE PLOT DATA (Per-Sample Scores)
# =============================================================================

def silhouette_per_sample(X: pd.DataFrame,
                            n_clusters: int = 3,
                            scale: bool = True,
                            random_state: int = 42) -> pd.DataFrame:
    """
    Computes per-sample silhouette scores for a fixed K —
    reveals which clusters are well-separated and which have
    overlapping or misassigned points.

    Args:
        X            : Feature DataFrame
        n_clusters   : Number of clusters
        scale        : Whether to StandardScale
        random_state : Reproducibility seed

    Returns:
        DataFrame with per-sample silhouette scores and cluster labels
    """
    X_proc = X.copy()
    if scale:
        X_proc = pd.DataFrame(
            StandardScaler().fit_transform(X), columns=X.columns
        )

    model  = KMeans(n_clusters=n_clusters, init="k-means++",
                     n_init=10, random_state=random_state)
    labels = model.fit_predict(X_proc)
    scores = silhouette_samples(X_proc, labels)

    df = pd.DataFrame({
        "Cluster"          : labels,
        "Silhouette Score" : scores.round(4),
    }).sort_values(["Cluster", "Silhouette Score"])

    overall = scores.mean()
    print(f"[Silhouette Per-Sample] n_clusters={n_clusters}")
    print(f"  Overall silhouette: {overall:.4f}")
    per_cluster = df.groupby("Cluster")["Silhouette Score"].mean().round(4)
    print(f"  Per-cluster mean silhouette:\n{per_cluster.to_string()}")
    return df


# =============================================================================
# 🔧 5. CLUSTER PROFILE — DESCRIBE EACH CLUSTER
# =============================================================================

def cluster_profile(X: pd.DataFrame,
                     labels: np.ndarray,
                     top_n: int = 5) -> pd.DataFrame:
    """
    Builds a statistical profile of each cluster —
    shows mean feature values per cluster to understand
    what makes each cluster distinct.

    Args:
        X      : Original (unscaled) feature DataFrame
        labels : Cluster assignment array from K-Means
        top_n  : Top N most discriminative features to highlight

    Returns:
        DataFrame with per-cluster feature statistics
    """
    X_prof = X.copy()
    X_prof["Cluster"] = labels

    profile = X_prof.groupby("Cluster").agg(["mean", "std", "count"])
    profile.columns = ["_".join(col) for col in profile.columns]

    print(f"[Cluster Profile] {labels.max()+1} clusters")
    mean_cols = [c for c in profile.columns if c.endswith("_mean")]
    print(profile[mean_cols].round(4).to_string())

    # Most discriminative features = highest variance of cluster means
    cluster_means = X_prof.groupby("Cluster").mean()
    feature_variance = cluster_means.var().sort_values(ascending=False)
    top_features = feature_variance.head(top_n)
    print(f"\n  Top {top_n} most discriminative features:")
    print(top_features.round(4).to_string())

    return profile


# =============================================================================
# 🔧 6. PREDICT CLUSTER FOR NEW DATA
# =============================================================================

def predict_cluster(model: KMeans,
                     scaler: StandardScaler,
                     X_new: pd.DataFrame) -> np.ndarray:
    """
    Assigns cluster labels to new, unseen data points using a
    fitted K-Means model and scaler.

    Args:
        model   : Fitted KMeans model
        scaler  : Fitted StandardScaler (or None if no scaling was used)
        X_new   : New feature DataFrame

    Returns:
        Array of cluster labels for each new point
    """
    X_proc = X_new.copy()
    if scaler is not None:
        X_proc = pd.DataFrame(
            scaler.transform(X_new), columns=X_new.columns
        )

    labels   = model.predict(X_proc)
    dists    = model.transform(X_proc).min(axis=1)

    print(f"[Predict Cluster] {len(X_new)} new samples assigned")
    pred_df = pd.DataFrame({
        "Cluster"          : labels,
        "Distance to Center": dists.round(4),
    })
    print(pred_df.value_counts("Cluster").sort_index().to_string())
    return labels


# =============================================================================
# 🔧 7. EVALUATE AGAINST GROUND TRUTH LABELS
# =============================================================================

def evaluate_with_labels(labels_pred: np.ndarray,
                           labels_true: np.ndarray,
                           model_name: str = "KMeans") -> pd.DataFrame:
    """
    Evaluates clustering quality using ground truth labels.

    These metrics can only be used when true labels are available —
    for purely unsupervised settings, use silhouette / inertia instead.

    Metrics:
        ARI  : Adjusted Rand Index — corrects for chance agreement
        NMI  : Normalized Mutual Information — information overlap
        Homo : Homogeneity — each cluster contains one class
        Comp : Completeness — all members of a class in one cluster
        V    : V-Measure — harmonic mean of Homo and Comp

    Args:
        labels_pred : Predicted cluster labels
        labels_true : True class labels
        model_name  : Name for display

    Returns:
        DataFrame with evaluation metrics
    """
    metrics = {
        "Model"         : model_name,
        "ARI"           : round(adjusted_rand_score(labels_true, labels_pred), 4),
        "NMI"           : round(normalized_mutual_info_score(labels_true, labels_pred), 4),
        "Homogeneity"   : round(homogeneity_score(labels_true, labels_pred), 4),
        "Completeness"  : round(completeness_score(labels_true, labels_pred), 4),
        "V-Measure"     : round(v_measure_score(labels_true, labels_pred), 4),
    }

    report = pd.DataFrame([metrics])
    print(f"\n📊 Cluster Evaluation (vs Ground Truth) — {model_name}")
    print(report.to_string(index=False))
    return report


# =============================================================================
# 🔧 8. MINI-BATCH K-MEANS (Large Datasets)
# =============================================================================

def train_minibatch_kmeans(X: pd.DataFrame,
                             n_clusters: int = 3,
                             batch_size: int = 1024,
                             n_init: int = 10,
                             scale: bool = True,
                             random_state: int = 42) -> dict:
    """
    Trains Mini-Batch K-Means — faster alternative for large datasets.

    Processes small random batches per iteration instead of full data:
        ✅ Much faster: O(batch_size) per iteration vs O(n)
        ✅ Lower memory usage
        ⚠️ Slightly worse clustering quality than full K-Means

    Recommended for n > 100,000 rows.

    Args:
        X            : Feature DataFrame
        n_clusters   : Number of clusters K
        batch_size   : Samples per mini-batch (default: 1024)
        n_init       : Number of random initializations
        scale        : Whether to StandardScale
        random_state : Reproducibility seed

    Returns:
        Dictionary with model, labels, inertia, and metrics
    """
    X_proc = X.copy()
    scaler = None

    if scale:
        scaler  = StandardScaler()
        X_proc  = pd.DataFrame(
            scaler.fit_transform(X), columns=X.columns
        )

    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        n_init=n_init,
        random_state=random_state,
    )
    model.fit(X_proc)

    labels  = model.labels_
    inertia = model.inertia_
    sil     = silhouette_score(X_proc, labels) if n_clusters > 1 else None

    print(f"[MiniBatchKMeans] n_clusters={n_clusters} | "
          f"batch_size={batch_size}")
    print(f"  Inertia     : {inertia:.4f}")
    print(f"  Silhouette  : {sil:.4f}" if sil else "")
    print(f"  Cluster sizes: "
          f"{pd.Series(labels).value_counts().sort_index().to_dict()}")

    return {
        "model"     : model,
        "scaler"    : scaler,
        "labels"    : labels,
        "X_scaled"  : X_proc,
        "inertia"   : inertia,
        "silhouette": sil,
    }


# =============================================================================
# 🔧 9. K-MEANS WITH PCA VISUALIZATION
# =============================================================================

def kmeans_with_pca(X: pd.DataFrame,
                     n_clusters: int = 3,
                     n_components: int = 2,
                     scale: bool = True,
                     random_state: int = 42) -> dict:
    """
    Runs K-Means and projects data to 2D via PCA for visualization.

    PCA compression allows plotting high-dimensional cluster results
    in 2D while preserving as much variance as possible.

    Args:
        X             : Feature DataFrame
        n_clusters    : Number of clusters K
        n_components  : PCA components (2 for 2D plot, 3 for 3D)
        scale         : Whether to StandardScale
        random_state  : Reproducibility seed

    Returns:
        Dictionary with labels, PCA-reduced data, and explained variance
    """
    X_proc = X.copy()
    scaler = None

    if scale:
        scaler  = StandardScaler()
        X_proc  = pd.DataFrame(
            scaler.fit_transform(X), columns=X.columns
        )

    # Cluster in original space
    model  = KMeans(n_clusters=n_clusters, init="k-means++",
                     n_init=10, random_state=random_state)
    labels = model.fit_predict(X_proc)

    # Reduce to n_components for visualization
    pca     = PCA(n_components=n_components, random_state=random_state)
    X_pca   = pca.fit_transform(X_proc)
    exp_var = pca.explained_variance_ratio_

    pca_cols = [f"PC{i+1}" for i in range(n_components)]
    X_pca_df = pd.DataFrame(X_pca, columns=pca_cols)
    X_pca_df["Cluster"] = labels

    sil = silhouette_score(X_proc, labels)

    print(f"[KMeans + PCA] n_clusters={n_clusters} | "
          f"n_components={n_components}")
    print(f"  Silhouette     : {sil:.4f}")
    print(f"  Explained var  : "
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
# 🔧 10. COMPARE K-MEANS VS MINI-BATCH
# =============================================================================

def compare_kmeans_variants(X: pd.DataFrame,
                              n_clusters: int = 3,
                              scale: bool = True) -> pd.DataFrame:
    """
    Compares standard K-Means vs Mini-Batch K-Means on quality metrics.

    Args:
        X          : Feature DataFrame
        n_clusters : Number of clusters K
        scale      : Whether to StandardScale

    Returns:
        DataFrame comparing inertia, silhouette, and timing
    """
    import time

    X_proc = X.copy()
    if scale:
        X_proc = pd.DataFrame(
            StandardScaler().fit_transform(X), columns=X.columns
        )

    rows = []
    for name, model in [
        ("KMeans",
         KMeans(n_clusters=n_clusters, n_init=10, random_state=42)),
        ("MiniBatchKMeans",
         MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024,
                          n_init=10, random_state=42)),
    ]:
        t0     = time.time()
        labels = model.fit_predict(X_proc)
        t1     = time.time()
        sil    = silhouette_score(X_proc, labels)
        ch     = calinski_harabasz_score(X_proc, labels)

        rows.append({
            "Model"            : name,
            "Inertia"          : round(model.inertia_, 4),
            "Silhouette"       : round(sil, 4),
            "Calinski-Harabasz": round(ch, 4),
            "Time (s)"         : round(t1 - t0, 4),
        })

    df = pd.DataFrame(rows)
    print(f"KMeans vs MiniBatchKMeans Comparison (K={n_clusters}):")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 HELPERS
# =============================================================================

def summarize_clusters(X: pd.DataFrame,
                        labels: np.ndarray) -> pd.DataFrame:
    """
    Quick summary of cluster sizes and basic statistics.
    """
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

    from sklearn.datasets import make_blobs, make_classification

    np.random.seed(42)

    # ── Synthetic clustered dataset ───────────────────────────────────────
    X_raw, y_true = make_blobs(
        n_samples=600,
        n_features=6,
        centers=4,
        cluster_std=1.2,
        random_state=42
    )
    X = pd.DataFrame(X_raw, columns=[f"Feature_{i+1}" for i in range(6)])

    print("=" * 65)
    print("📊 Dataset Info — Synthetic Blobs")
    print("=" * 65)
    print(f"Shape      : {X.shape}")
    print(f"True groups: 4  |  y_true unique: {np.unique(y_true)}")

    # ── 1. Basic K-Means (K=4) ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("1️⃣  K-Means (K=4)")
    print("=" * 65)
    result = train_kmeans(X, n_clusters=4)

    # ── 2. Elbow Method ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("2️⃣  Elbow Method")
    print("=" * 65)
    elbow_df = elbow_method(X, k_range=range(1, 12))

    # ── 3. Silhouette Analysis ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("3️⃣  Silhouette Analysis")
    print("=" * 65)
    sil_df = silhouette_analysis(X, k_range=range(2, 12))

    # ── 4. Per-Sample Silhouette ──────────────────────────────────────────
    print("\n" + "=" * 65)
    print("4️⃣  Per-Sample Silhouette Scores")
    print("=" * 65)
    sample_sil = silhouette_per_sample(X, n_clusters=4)

    # ── 5. Cluster Profile ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("5️⃣  Cluster Profile")
    print("=" * 65)
    profile = cluster_profile(X, result["labels"])

    # ── 6. Evaluate with Ground Truth ─────────────────────────────────────
    print("\n" + "=" * 65)
    print("6️⃣  Evaluate Against True Labels")
    print("=" * 65)
    eval_df = evaluate_with_labels(result["labels"], y_true)

    # ── 7. KMeans + PCA ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("7️⃣  K-Means + PCA (2D projection)")
    print("=" * 65)
    pca_result = kmeans_with_pca(X, n_clusters=4, n_components=2)
    print(pca_result["X_pca"].head())

    # ── 8. Mini-Batch K-Means ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("8️⃣  Mini-Batch K-Means")
    print("=" * 65)
    mb_result = train_minibatch_kmeans(X, n_clusters=4)

    # ── 9. Compare Variants ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("9️⃣  KMeans vs MiniBatchKMeans")
    print("=" * 65)
    compare_df = compare_kmeans_variants(X, n_clusters=4)

    # ── 10. Predict new points ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("🔟  Predict Cluster for New Data")
    print("=" * 65)
    X_new = pd.DataFrame(
        np.random.randn(5, 6),
        columns=[f"Feature_{i+1}" for i in range(6)]
    )
    new_labels = predict_cluster(result["model"], result["scaler"], X_new)

    print("\n✅ All K-Means techniques demonstrated successfully!")
