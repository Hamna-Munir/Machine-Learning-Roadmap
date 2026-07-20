# 📘 DBSCAN — Theory

---

## 📌 What is DBSCAN?

DBSCAN (**Density-Based Spatial Clustering of Applications with Noise**) is a  
**density-based clustering algorithm** that groups together points that are  
closely packed (high density), and marks points in low-density regions as  
**outliers/noise** — without requiring the number of clusters in advance.

```
Idea: A cluster is a dense region of points separated from other
      dense regions by areas of low density.

Key property: Can find clusters of ARBITRARY SHAPE
              AND automatically detect OUTLIERS
```

**Visual intuition:**
```
Before DBSCAN:            After DBSCAN:
  ●●●●   ●●●●              ■■■■   ▲▲▲▲       ■ = Cluster 1
 ●●●●●  ●●●●●             ■■■■■  ▲▲▲▲▲      ▲ = Cluster 2
  ●●      ●●               ■■      ▲▲         ✕ = Noise (outliers)
      ●                        ✕
        ●  ●                     ✕  ✕
   ◆◆◆◆◆◆◆◆◆              ◆◆◆◆◆◆◆◆◆          ◆ = Cluster 3
  ◆◆◆◆◆◆◆◆◆◆             ◆◆◆◆◆◆◆◆◆◆
```

> 💡 "DBSCAN doesn't care how many clusters there are or what shape they are.  
>      It simply asks: 'Is this point in a dense neighborhood or not?'"

---

## 🔍 When to Use DBSCAN?

| Condition | Use DBSCAN? |
|-----------|:-----------:|
| Clusters have arbitrary/non-convex shapes | ✅ Yes — primary strength |
| Number of clusters unknown | ✅ Yes — no K needed |
| Data contains outliers/noise to detect | ✅ Yes — labels them as -1 |
| Clusters have similar density | ✅ Yes |
| Very large datasets | ⚠️ O(n log n) with KD-Tree |
| Clusters have very different densities | ❌ No → HDBSCAN |
| High-dimensional data | ❌ No → curse of dimensionality |
| Data is uniformly distributed | ❌ No — no density contrast |

---

## 🧮 Core Concepts

### Two Hyperparameters

```
ε (eps)          : Radius of the neighborhood around a point
                   "How far to look for neighbors?"

min_samples      : Minimum number of points required within ε
                   to consider a region "dense"
                   "How many neighbors needed to form a dense region?"
```

### Three Point Types

```
For each point p:
  N_ε(p) = {q : d(p, q) ≤ ε}   (ε-neighborhood of p)
  |N_ε(p)| = number of points within ε of p

Core Point:     |N_ε(p)| ≥ min_samples
                → p is in a dense region → PART OF A CLUSTER

Border Point:   |N_ε(p)| < min_samples  AND
                ∃ core point c such that d(p, c) ≤ ε
                → p is reachable from a core point → PART OF A CLUSTER

Noise Point:    Neither core nor border
                → p is in a sparse region → LABELED AS OUTLIER (-1)
```

**Visual:**
```
                    ●    ← core point (many neighbors)
                  ●●●●
                ●●●●●●●
                  ●●●    ← core points
                 ●  ●    ← border points (inside ε of a core)
                          ●    ← noise (too far from any core)
                   ●         ← noise
```

---

## 🔄 The Algorithm

```
DBSCAN(X, ε, min_samples):

  Label all points as UNVISITED
  cluster_id = 0

  For each unvisited point p:
    Mark p as VISITED

    neighbors = RangeQuery(X, p, ε)   ← find all points within ε of p

    If |neighbors| < min_samples:
      Label p as NOISE (-1)            ← sparse region

    Else:                              ← p is a CORE POINT
      cluster_id += 1
      Assign p to cluster_id
      seed_set = neighbors

      While seed_set is not empty:
        q = seed_set.pop()

        If q is NOISE:
          Assign q to cluster_id      ← border point absorbed

        If q is UNVISITED:
          Mark q as VISITED
          q_neighbors = RangeQuery(X, q, ε)

          If |q_neighbors| ≥ min_samples:   ← q is also a core point
            seed_set = seed_set ∪ q_neighbors  ← expand cluster

          Assign q to cluster_id

  Return labels (-1 = noise, 0..K-1 = cluster IDs)
```

**Complexity:**
```
With spatial index (KD-Tree / Ball-Tree):
  Time:   O(n log n)
  Space:  O(n)

Brute force (no index):
  Time:   O(n²)
  Space:  O(n²)

sklearn default: algorithm='auto' → uses KD-Tree for low-d, Ball-Tree otherwise
```

---

## 🎛️ Choosing ε and min_samples

### Choosing min_samples

```
Rule of thumb:
  min_samples ≥ dimensionality + 1    (for low-noise data)
  min_samples ≥ 2 × dimensionality    (for noisy data)

Typical values: 4–10

Higher min_samples → stricter density requirement → more noise points
Lower min_samples  → looser density → fewer noise points, larger clusters
```

### Choosing ε — k-Distance Plot

```
Best practice to choose ε:

1. For each point, compute distance to its k-th nearest neighbor
   (k = min_samples − 1)

2. Sort these distances in descending order

3. Plot sorted k-distances

4. ε = value at the "knee" / "elbow" of the curve

k-Distance
│●
│ ●
│  ●●
│    ●●●
│       ●●●●●●          ← knee/elbow here
│              ●●●●●●●●●
└──────────────────────── Points (sorted)

from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X)
distances, _ = nbrs.kneighbors(X)
k_distances  = np.sort(distances[:, -1])[::-1]
```

---

## 📐 Density Reachability Concepts

```
Direct density-reachable:
  p is directly density-reachable from q if:
    q is a core point AND d(p, q) ≤ ε

Density-reachable:
  p is density-reachable from q if there exists a chain:
    q = x₁, x₂, ..., xₙ = p  where each xᵢ₊₁ is directly reachable from xᵢ

Density-connected:
  p and q are density-connected if ∃ core point o such that:
    both p and q are density-reachable from o

Cluster definition:
  C is a valid cluster if:
    1. All points in C are mutually density-connected
    2. C contains all density-reachable points from any of its core points
```

---

## 🔢 Output — Cluster Labels

```
DBSCAN labels:
  -1  → Noise / Outlier
   0  → Cluster 0
   1  → Cluster 1
  ...

sklearn: model.labels_
         model.core_sample_indices_   ← indices of core points

Number of clusters = len(set(labels)) - (1 if -1 in labels else 0)
Number of noise    = (labels == -1).sum()
```

---

## 🌟 HDBSCAN — Hierarchical DBSCAN

An extension that handles **clusters of varying density**:

```
Standard DBSCAN limitation:
  Uses single global ε → fails when clusters have different densities

HDBSCAN solution:
  Builds a hierarchy of clusters at all density levels
  → Extracts stable clusters using a stability criterion
  → Handles multi-density data automatically

sklearn: from sklearn.cluster import HDBSCAN
         model = HDBSCAN(min_cluster_size=5)
```

---

## 🎛️ Key Hyperparameters

| Parameter | Effect | Typical Values |
|-----------|--------|----------------|
| `eps` | Neighborhood radius ε | Find via k-distance plot |
| `min_samples` | Min points for dense region | d+1 to 2d (d=dimensions) |
| `metric` | Distance function | 'euclidean' (default), 'manhattan', 'cosine' |
| `algorithm` | Neighbor search | 'auto', 'ball_tree', 'kd_tree', 'brute' |
| `leaf_size` | KD/Ball tree leaf size | 30 (default) — affects speed |
| `n_jobs` | Parallel threads | -1 (all cores) |

---

## 🆚 DBSCAN vs K-Means vs Hierarchical

| Aspect | DBSCAN | K-Means | Hierarchical |
|--------|:------:|:-------:|:------------:|
| Need K in advance | ❌ No | ✅ Yes | ❌ No |
| Finds outliers | ✅ Yes (-1) | ❌ No | ❌ No |
| Arbitrary shapes | ✅ Yes | ❌ Spherical | ✅ (single link) |
| Varying density | ❌ No | ❌ No | ❌ No |
| Deterministic | ✅ Yes | ❌ No | ✅ Yes |
| Scalability | ✅ O(n log n) | ✅ O(nKI) | ❌ O(n²) |
| Feature scaling | ✅ Required | ✅ Required | ✅ Required |
| Hyperparameter sensitivity | ⚠️ High | ⚠️ Medium | ⚠️ Low |

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Not scaling features | ε not meaningful across different scales | Always StandardScale first |
| Wrong ε choice | All noise or one giant cluster | Use k-distance plot to find knee |
| Wrong min_samples | Too many/few noise points | Rule: min_samples ≥ dimensionality + 1 |
| High-dimensional data | All points equidistant — no density contrast | Apply PCA first |
| Clusters of varying density | Single ε fails | Use HDBSCAN instead |
| Ignoring noise points | Miss important signal | Check noise fraction and investigate |

---

## 🔗 Related Topics

- `KMeans` — Partitioning alternative (requires K, spherical clusters)
- `Hierarchical_Clustering` — Tree-based alternative (no K needed)
- `04_Unsupervised_Learning/PCA` — Reduce dimensions before DBSCAN
- `05_Model_Evaluation` — Silhouette score, ARI for cluster evaluation

---

## 📚 References

- Scikit-learn `DBSCAN`: [https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- Scikit-learn `HDBSCAN`: [https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html)
- Original DBSCAN Paper (Ester et al., 1996): [https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf](https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf)
- An Introduction to Statistical Learning — Chapter 12.4
- The Elements of Statistical Learning — Chapter 14.3
