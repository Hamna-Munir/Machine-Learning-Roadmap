# =============================================================================
# 📦 DBSCAN — Reusable ML Script
# =============================================================================
# Author   : Hamna Munir
# Topic    : 04_Unsupervised_Learning / DBSCAN
# File     : dbscan.py
# =============================================================================

# -----------------------------------------------------------------------------
# 📚 Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score
)

import warnings
warnings.filterwarnings("ignore")

# Optional: HDBSCAN (sklearn >= 1.3)
try:
    from sklearn.cluster import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


# =============================================================================
# 🔧 1. TRAIN DBSCAN
# =============================================================================

def train_dbscan(X: pd.DataFrame,
                  eps: float = 0.5,
                  min_samples: int = 5,
                  metric: str = "euclidean",
                  algorithm: str = "auto",
                  scale: bool = True) -> dict:
    """
    Trains a DBSCAN clustering model.

    Algorithm:
        For each point p:
          If |N_ε(p)| ≥ min_samples → CORE POINT → start/expand cluster
          If reachable from a core   → BORDER POINT → join cluster
          Otherwise                 → NOISE → label = -1

    ⚠️ Always scale features before DBSCAN — ε is distance-based
        and meaningless without standardized feature ranges.

    Args:
        X          : Feature DataFrame (no labels)
        eps        : Neighborhood radius ε
                     → Too small: everything is noise
                     → Too large: everything in one cluster
                     → Use k_distance_plot() to find optimal ε
        min_samples: Minimum neighbors within ε to be a core point
                     → Rule: ≥ dimensionality + 1 (noisy data: ≥ 2×dim)
        metric     : Distance metric — 'euclidean', 'manhattan', 'cosine'
        algorithm  : Neighbor search — 'auto', 'ball_tree', 'kd_tree', 'brute'
        scale      : Whether to StandardScale features (default: True)

    Returns:
        Dictionary with model, labels, core indices, and metrics
    """
    X_proc = X.copy()
    scaler = None

    if scale:
        scaler  = StandardScaler()
        X_proc  = pd.DataFrame(
            scaler.fit_transform(X), columns=X.columns
        )

    model  = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        algorithm=algorithm,
        n_jobs=-1,
    )
    labels = model.fit_predict(X_proc)

    n_clusters  = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise     = (labels == -1).sum()
    noise_pct   = n_noise / len(X) * 100
    core_idx    = model.core_sample_indices_
    cluster_sizes = (
        pd.Series(labels[labels != -1]).value_counts().sort_index()
        if n_clusters > 0 else pd.Series(dtype=int)
    )

    # Metrics only valid when ≥ 2 clusters and not all noise
    sil_score = None
    ch_score  = None
    db_score  = None
    if n_clusters >= 2:
        mask      = labels != -1
        sil_score = silhouette_score(X_proc[mask], labels[mask])
        ch_score  = calinski_harabasz_score(X_proc, labels)
        db_score  = davies_bouldin_score(X_proc[mask], labels[mask])

    print(f"[DBSCAN] eps={eps} | min_samples={min_samples} | "
          f"metric={metric}")
    print(f"  Clusters found  : {n_clusters}")
    print(f"  Noise points    : {n_noise} ({noise_pct:.1f}%)")
    print(f"  Core points     : {len(core_idx)}")
    if n_clusters > 0:
        print(f"  Cluster sizes   : {cluster_sizes.to_dict()}")
    if sil_score is not None:
        print(f"  Silhouette Score: {sil_score:.4f}")
        print(f"  Calinski-Harabasz: {ch_score:.4f}")
        print(f"  Davies-Bouldin  : {db_score:.4f}")

    return {
        "model"         : model,
        "scaler"        : scaler,
        "labels"        : labels,
        "X_scaled"      : X_proc,
        "n_clusters"    : n_clusters,
        "n_noise"       : n_noise,
        "noise_pct"     : noise_pct,
        "core_indices"  : core_idx,
        "cluster_sizes" : cluster_sizes,
        "silhouette"    : sil_score,
        "calinski"      : ch_score,
        "davies_bouldin": db_score,
    }


# =============================================================================
# 🔧 2. K-DISTANCE PLOT — FIND OPTIMAL EPS
# =============================================================================

def k_distance_plot(X: pd.DataFrame,
                     min_samples: int = 5,
                     scale: bool = True) -> np.ndarray:
    """
    Computes k-distances to help choose the optimal ε (eps) value.

    Method:
        1. For each point, compute distance to its k-th nearest neighbor
           (k = min_samples)
        2. Sort distances in descending order
        3. The "knee" / "elbow" of the curve is the optimal ε

    The knee represents the transition between:
        - Dense regions (small distances, below the knee)
        - Sparse regions / noise (large distances, above the knee)

    Args:
        X           : Feature DataFrame
        min_samples : k value for k-th nearest neighbor
                      (should match DBSCAN min_samples)
        scale       : Whether to StandardScale

    Returns:
        Sorted k-distances array (descending)
    """
    X_proc = X.copy()
    if scale:
        X_proc = pd.DataFrame(
            StandardScaler().fit_transform(X), columns=X.columns
        )

    nbrs = NearestNeighbors(n_neighbors=min_samples, n_jobs=-1)
    nbrs.fit(X_proc)
    distances, _ = nbrs.kneighbors(X_proc)

    k_distances = np.sort(distances[:, -1])[::-1]

    # Find approximate knee (point of maximum curvature)
    diffs      = np.diff(k_distances)
    knee_idx   = np.argmin(diffs) + 1
    knee_eps   = k_distances[knee_idx]

    print(f"[k-Distance Plot] k={min_samples}")
    print(f"  Distance range   : [{k_distances.min():.4f}, "
          f"{k_distances.max():.4f}]")
    print(f"  Knee index       : {knee_idx}")
    print(f"  Suggested ε (eps): {knee_eps:.4f}")
    print(f"  → Try eps values around {knee_eps:.2f} ± 20%")

    return k_distances


# =============================================================================
# 🔧 3. EPS SENSITIVITY ANALYSIS
# =============================================================================

def eps_sensitivity(X: pd.DataFrame,
                     eps_values: list = None,
                     min_samples: int = 5,
                     scale: bool = True) -> pd.DataFrame:
    """
    Evaluates DBSCAN across a range of ε (eps) values —
    shows how the number of clusters and noise points change.

    Too small ε → everything is noise (1 point clusters)
    Too large ε → everything merges into one cluster

    Args:
        X           : Feature DataFrame
        eps_values  : List of eps values to evaluate
        min_samples : Fixed min_samples for comparison
        scale       : Whether to StandardScale

    Returns:
        DataFrame with n_clusters, n_noise, and silhouette per eps
    """
    if eps_values is None:
        eps_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]

    X_proc = X.copy()
    if scale:
        X_proc = pd.DataFrame(
            StandardScaler().fit_transform(X), columns=X.columns
        )

    rows = []
    for eps in eps_values:
        model   = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels  = model.fit_predict(X_proc)
        n_clust = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        mask    = labels != -1

        sil = None
        if n_clust >= 2 and mask.sum() > n_clust:
            try:
                sil = round(silhouette_score(X_proc[mask], labels[mask]), 4)
            except Exception:
                pass

        rows.append({
            "eps"           : eps,
            "n_clusters"    : n_clust,
            "n_noise"       : n_noise,
            "noise_%"       : round(n_noise / len(X) * 100, 1),
            "Silhouette"    : sil if sil else "N/A",
        })

    df = pd.DataFrame(rows)
    print(f"eps Sensitivity (min_samples={min_samples}):")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 4. MIN_SAMPLES SENSITIVITY ANALYSIS
# =============================================================================

def min_samples_sensitivity(X: pd.DataFrame,
                              eps: float = 0.5,
                              min_samples_values: list = None,
                              scale: bool = True) -> pd.DataFrame:
    """
    Evaluates DBSCAN across a range of min_samples values —
    shows how density threshold affects cluster structure.

    Higher min_samples → stricter density → more noise points
    Lower min_samples  → looser density   → fewer noise points

    Args:
        X                  : Feature DataFrame
        eps                : Fixed eps for comparison
        min_samples_values : List of min_samples to evaluate
        scale              : Whether to StandardScale

    Returns:
        DataFrame with n_clusters, n_noise per min_samples
    """
    if min_samples_values is None:
        min_samples_values = [2, 3, 4, 5, 7, 10, 15, 20]

    X_proc = X.copy()
    if scale:
        X_proc = pd.DataFrame(
            StandardScaler().fit_transform(X), columns=X.columns
        )

    rows = []
    for ms in min_samples_values:
        model   = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1)
        labels  = model.fit_predict(X_proc)
        n_clust = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        mask    = labels != -1

        sil = None
        if n_clust >= 2 and mask.sum() > n_clust:
            try:
                sil = round(silhouette_score(X_proc[mask], labels[mask]), 4)
            except Exception:
                pass

        rows.append({
            "min_samples": ms,
            "n_clusters" : n_clust,
            "n_noise"    : n_noise,
            "noise_%"    : round(n_noise / len(X) * 100, 1),
            "n_core"     : len(model.core_sample_indices_),
            "Silhouette" : sil if sil else "N/A",
        })

    df = pd.DataFrame(rows)
    print(f"min_samples Sensitivity (eps={eps}):")
    print(df.to_string(index=False))
    return df


# =============================================================================
# 🔧 5. GRID SEARCH — EPS × MIN_SAMPLES
# =============================================================================

def grid_search_dbscan(X: pd.DataFrame,
                        eps_values: list = None,
                        min_samples_values: list = None,
                        scale: bool = True) -> pd.DataFrame:
    """
    Exhaustive grid search over eps × min_samples combinations.
    Ranks by silhouette score while filtering degenerate results.

    Only considers parameter combinations that yield:
        - At least 2 clusters
        - At most 50% noise points
        - At least 10 non-noise points

    Args:
        X                  : Feature DataFrame
        eps_values         : List of eps values
        min_samples_values : List of min_samples values
        scale              : Whether to StandardScale

    Returns:
        DataFrame sorted by silhouette score (best first)
    """
    if eps_values is None:
        eps_values = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
    if min_samples_values is None:
        min_samples_values = [3, 4, 5, 7, 10]

    X_proc = X.copy()
    if scale:
        X_proc = pd.DataFrame(
            StandardScaler().fit_transform(X), columns=X.columns
        )

    rows = []
    for eps in eps_values:
        for ms in min_samples_values:
            model   = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1)
            labels  = model.fit_predict(X_proc)
            n_clust = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            mask    = labels != -1

            if n_clust < 2:
                continue
            if n_noise / len(X) > 0.5:
                continue
            if mask.sum() < 10:
                continue

            try:
                sil = silhouette_score(X_proc[mask], labels[mask])
            except Exception:
                continue

            rows.append({
                "eps"       : eps,
                "min_samples": ms,
                "n_clusters": n_clust,
                "n_noise"   : n_noise,
                "noise_%"   : round(n_noise / len(X) * 100, 1),
                "Silhouette": round(sil, 4),
            })

    if not rows:
        print("No valid parameter combinations found.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(
        "Silhouette", ascending=False
    ).reset_index(drop=True)

    print("DBSCAN Grid Search Results (sorted by silhouette):")
    print(df.head(10).to_string(index=False))
    print(f"\n  Best eps={df.iloc[0]['eps']} | "
          f"min_samples={df.iloc[0]['min_samples']} | "
          f"Silhouette={df.iloc[0]['Silhouette']}")
    return df


# =============================================================================
# 🔧 6. TRAIN HDBSCAN (Hierarchical DBSCAN)
# =============================================================================

def train_hdbscan(X: pd.DataFrame,
                   min_cluster_size: int = 5,
                   min_samples: int = None,
                   metric: str = "euclidean",
                   scale: bool = True) -> dict:
    """
    Trains HDBSCAN — handles clusters of varying density.

    HDBSCAN builds a hierarchy of clusters at all density levels
    and extracts stable clusters automatically.

    Advantages over DBSCAN:
        ✅ No ε parameter needed
        ✅ Handles clusters of different densities
        ✅ More robust parameter choices

    Args:
        X                : Feature DataFrame
        min_cluster_size : Minimum cluster size (most important param)
                           Typical: 5–50 depending on dataset size
        min_samples      : Controls conservatism (None = min_cluster_size)
                           Higher → more conservative → more noise
        metric           : Distance metric
        scale            : Whether to StandardScale

    Returns:
        Dictionary with model, labels, and metrics
    """
    if not HDBSCAN_AVAILABLE:
        print("⚠️  HDBSCAN requires scikit-learn >= 1.3")
        print("    Run: pip install --upgrade scikit-learn")
        return {}

    X_proc = X.copy()
    scaler = None

    if scale:
        scaler  = StandardScaler()
        X_proc  = pd.DataFrame(
            scaler.fit_transform(X), columns=X.columns
        )

    model  = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )
    labels = model.fit_predict(X_proc)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    mask       = labels != -1

    sil_score = None
    if n_clusters >= 2 and mask.sum() > n_clusters:
        sil_score = silhouette_score(X_proc[mask], labels[mask])

    print(f"[HDBSCAN] min_cluster_size={min_cluster_size} | "
          f"min_samples={min_samples}")
    print(f"  Clusters found  : {n_clusters}")
    print(f"  Noise points    : {n_noise} ({n_noise/len(X)*100:.1f}%)")
    if sil_score:
        print(f"  Silhouette Score: {sil_score:.4f}")

    return {
        "model"      : model,
        "scaler"     : scaler,
        "labels"     : labels,
        "X_scaled"   : X_proc,
        "n_clusters" : n_clusters,
        "n_noise"    : n_noise,
        "silhouette" : sil_score,
    }


# =============================================================================
# 🔧 7. OUTLIER / NOISE ANALYSIS
# =============================================================================

def analyze_noise(X: pd.DataFrame,
                   labels: np.ndarray) -> pd.DataFrame:
    """
    Extracts and profiles the noise points detected by DBSCAN.

    Noise points (-1 labels) represent genuine outliers —
    they are in sparse regions and not reachable from any core point.

    Args:
        X      : Original (unscaled) feature DataFrame
        labels : DBSCAN cluster labels

    Returns:
        DataFrame of noise points with their feature values
    """
    noise_mask  = labels == -1
    n_noise     = noise_mask.sum()
    n_total     = len(X)

    print(f"[Noise Analysis]")
    print(f"  Total points : {n_total}")
    print(f"  Noise points : {n_noise} ({n_noise/n_total*100:.1f}%)")
    print(f"  Cluster pts  : {n_total - n_noise}")

    if n_noise == 0:
        print("  No noise points detected.")
        return pd.DataFrame()

    noise_df    = X[noise_mask].copy()
    cluster_df  = X[~noise_mask].copy()

    print(f"\n  Noise point statistics:")
    print(noise_df.describe().round(4).to_string())

    # Compare noise vs cluster means
    comparison = pd.DataFrame({
        "Noise Mean"  : noise_df.mean(),
        "Cluster Mean": cluster_df.mean() if len(cluster_df) > 0
                        else pd.Series(np.nan, index=X.columns),
        "Difference"  : noise_df.mean() - (
            cluster_df.mean() if len(cluster_df) > 0
            else pd.Series(0, index=X.columns)
        ),
    }).round(4)

    print(f"\n  Noise vs Cluster Means:")
    print(comparison.to_string())

    return noise_df


# =============================================================================
# 🔧 8. CLUSTER PROFILE
# =============================================================================

def cluster_profile(X: pd.DataFrame,
                     labels: np.ndarray,
                     include_noise: bool = False) -> pd.DataFrame:
    """
    Builds a statistical profile of each cluster.

    Args:
        X             : Original feature DataFrame
        labels        : DBSCAN cluster labels
        include_noise : Whether to include noise (-1) in the profile

    Returns:
        DataFrame with per-cluster feature statistics
    """
    X_prof = X.copy()
    X_prof["Cluster"] = labels

    if not include_noise:
        X_prof = X_prof[X_prof["Cluster"] != -1]

    means  = X_prof.groupby("Cluster").mean().round(4)
    counts = X_prof.groupby("Cluster").size().rename("Count")
    profile = pd.concat([counts, means], axis=1)

    print(f"[Cluster Profile] (noise excluded: {not include_noise})")
    print(profile.to_string())
    return profile


# =============================================================================
# 🔧 9. DBSCAN WITH PCA VISUALIZATION
# =============================================================================

def dbscan_with_pca(X: pd.DataFrame,
                     eps: float = 0.5,
                     min_samples: int = 5,
                     n_components: int = 2,
                     scale: bool = True) -> dict:
    """
    Runs DBSCAN and projects to 2D/3D via PCA for visualization.

    Args:
        X            : Feature DataFrame
        eps          : DBSCAN epsilon
        min_samples  : DBSCAN min_samples
        n_components : PCA output dimensions (2 or 3)
        scale        : Whether to StandardScale

    Returns:
        Dictionary with labels, PCA-reduced DataFrame, and metrics
    """
    X_proc = X.copy()
    scaler = None

    if scale:
        scaler  = StandardScaler()
        X_proc  = pd.DataFrame(
            scaler.fit_transform(X), columns=X.columns
        )

    model  = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = model.fit_predict(X_proc)

    pca     = PCA(n_components=n_components, random_state=42)
    X_pca   = pca.fit_transform(X_proc)
    exp_var = pca.explained_variance_ratio_

    pca_cols = [f"PC{i+1}" for i in range(n_components)]
    X_pca_df = pd.DataFrame(X_pca, columns=pca_cols)
    X_pca_df["Cluster"] = labels
    X_pca_df["IsNoise"] = labels == -1

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()

    print(f"[DBSCAN + PCA] eps={eps} | min_samples={min_samples}")
    print(f"  Clusters     : {n_clusters}")
    print(f"  Noise points : {n_noise}")
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
        "n_clusters"  : n_clusters,
        "n_noise"     : n_noise,
    }


# =============================================================================
# 🔧 10. EVALUATE AGAINST GROUND TRUTH
# =============================================================================

def evaluate_with_labels(labels_pred: np.ndarray,
                           labels_true: np.ndarray,
                           model_name: str = "DBSCAN") -> pd.DataFrame:
    """
    Evaluates DBSCAN clustering quality using ground truth labels.

    Note: Noise points (-1) are excluded from external metrics
    since they don't belong to any cluster.

    Args:
        labels_pred : Predicted cluster labels (-1 = noise)
        labels_true : True class labels
        model_name  : Name for display

    Returns:
        DataFrame with ARI, NMI, Homogeneity, Completeness, V-Measure
    """
    # Exclude noise points for external metrics
    mask        = labels_pred != -1
    lp_filtered = labels_pred[mask]
    lt_filtered = labels_true[mask] if hasattr(labels_true, '__len__') \
                  else labels_true

    if len(np.unique(lp_filtered)) < 2:
        print(f"⚠️  Only {len(np.unique(lp_filtered))} cluster(s) after "
              f"excluding noise — metrics may be unreliable.")

    metrics = {
        "Model"       : model_name,
        "ARI"         : round(adjusted_rand_score(
                            lt_filtered, lp_filtered), 4),
        "NMI"         : round(normalized_mutual_info_score(
                            lt_filtered, lp_filtered), 4),
        "Homogeneity" : round(homogeneity_score(
                            lt_filtered, lp_filtered), 4),
        "Completeness": round(completeness_score(
                            lt_filtered, lp_filtered), 4),
        "V-Measure"   : round(v_measure_score(
                            lt_filtered, lp_filtered), 4),
        "Noise Points": int((labels_pred == -1).sum()),
    }

    report = pd.DataFrame([metrics])
    print(f"\n📊 Cluster Evaluation (vs Ground Truth) — {model_name}")
    print(f"  (noise points excluded from metrics)")
    print(report.to_string(index=False))
    return report


# =============================================================================
# 🔧 HELPERS
# =============================================================================

def summarize_dbscan(labels: np.ndarray, X: pd.DataFrame) -> None:
    """Quick summary of DBSCAN results."""
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"[DBSCAN Summary]")
    print(f"  n_samples  : {len(X)}")
    print(f"  n_clusters : {n_clusters}")
    print(f"  n_noise    : {n_noise} ({n_noise/len(X)*100:.1f}%)")
    if n_clusters > 0:
        sizes = pd.Series(labels[labels != -1]).value_counts().sort_index()
        print(f"  Cluster sizes: {sizes.to_dict()}")


# =============================================================================
# 🚀 MAIN — Demo with Synthetic Dataset
# =============================================================================

if __name__ == "__main__":

    from sklearn.datasets import make_blobs, make_moons, make_circles

    np.random.seed(42)

    # ── Dataset 1: Non-convex shapes (moons) ──────────────────────────────
    X_moons, y_moons = make_moons(n_samples=400, noise=0.08, random_state=42)
    X_moons = pd.DataFrame(X_moons, columns=["Feature_1", "Feature_2"])

    # ── Dataset 2: Blobs with outliers ────────────────────────────────────
    X_blobs_raw, y_blobs = make_blobs(
        n_samples=400, n_features=4, centers=3,
        cluster_std=0.8, random_state=42
    )
    # Add manual outliers
    outliers = np.random.uniform(low=-8, high=8, size=(20, 4))
    X_blobs_raw = np.vstack([X_blobs_raw, outliers])
    y_blobs     = np.concatenate([y_blobs, [-1]*20])
    X_blobs = pd.DataFrame(
        X_blobs_raw, columns=[f"Feature_{i+1}" for i in range(4)]
    )

    print("=" * 65)
    print("📊 Dataset 1 — Moons (non-convex shapes)")
    print("=" * 65)
    print(f"Shape: {X_moons.shape}")

    # ── 1. K-Distance Plot (find eps) ─────────────────────────────────────
    print("\n" + "=" * 65)
    print("1️⃣  K-Distance Plot — Find Optimal eps")
    print("=" * 65)
    k_dists = k_distance_plot(X_moons, min_samples=5)

    # ── 2. DBSCAN on Moons ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("2️⃣  DBSCAN on Moon Dataset")
    print("=" * 65)
    result_moons = train_dbscan(X_moons, eps=0.2, min_samples=5)

    # ── 3. eps Sensitivity ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("3️⃣  eps Sensitivity Analysis")
    print("=" * 65)
    eps_df = eps_sensitivity(
        X_moons,
        eps_values=[0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0],
        min_samples=5
    )

    # ── 4. min_samples Sensitivity ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("4️⃣  min_samples Sensitivity Analysis")
    print("=" * 65)
    ms_df = min_samples_sensitivity(X_moons, eps=0.2)

    # ── 5. Grid Search ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("5️⃣  Grid Search — eps × min_samples")
    print("=" * 65)
    grid_df = grid_search_dbscan(
        X_moons,
        eps_values=[0.1, 0.15, 0.2, 0.25, 0.3],
        min_samples_values=[3, 4, 5, 7, 10]
    )

    print("\n" + "=" * 65)
    print("📊 Dataset 2 — Blobs with Outliers")
    print("=" * 65)
    print(f"Shape: {X_blobs.shape} (includes 20 manual outliers)")

    # ── 6. DBSCAN on Blobs ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("6️⃣  DBSCAN on Blobs with Outliers")
    print("=" * 65)
    result_blobs = train_dbscan(X_blobs, eps=1.0, min_samples=5)

    # ── 7. Noise Analysis ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("7️⃣  Noise / Outlier Analysis")
    print("=" * 65)
    noise_df = analyze_noise(X_blobs, result_blobs["labels"])

    # ── 8. Cluster Profile ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("8️⃣  Cluster Profile")
    print("=" * 65)
    profile = cluster_profile(X_blobs, result_blobs["labels"])

    # ── 9. Evaluate vs Ground Truth ───────────────────────────────────────
    print("\n" + "=" * 65)
    print("9️⃣  Evaluate vs Ground Truth")
    print("=" * 65)
    # Note: y_blobs has -1 for manual outliers; filter for comparison
    true_mask  = y_blobs != -1
    eval_df = evaluate_with_labels(
        result_blobs["labels"][true_mask],
        y_blobs[true_mask].astype(int)
    )

    # ── 10. DBSCAN + PCA ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("🔟  DBSCAN + PCA Projection")
    print("=" * 65)
    pca_result = dbscan_with_pca(X_blobs, eps=1.0, min_samples=5)

    # ── 11. HDBSCAN (if available) ────────────────────────────────────────
    if HDBSCAN_AVAILABLE:
        print("\n" + "=" * 65)
        print("1️⃣1️⃣  HDBSCAN (Varying Density)")
        print("=" * 65)
        hdbscan_result = train_hdbscan(X_blobs, min_cluster_size=10)
    else:
        print("\n⚠️  HDBSCAN skipped — requires sklearn >= 1.3")

    print("\n✅ All DBSCAN techniques demonstrated successfully!")
