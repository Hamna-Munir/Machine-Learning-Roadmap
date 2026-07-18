# 📘 K-Means Clustering — Theory

---

## 📌 What is K-Means Clustering?

K-Means is an **unsupervised learning algorithm** that partitions a dataset  
into **K distinct, non-overlapping clusters** by iteratively assigning each  
data point to the nearest cluster centroid and updating the centroids  
until convergence.

```
Input:  n data points, hyperparameter K (number of clusters)
Output: K cluster assignments + K centroids (cluster centers)

No labels required — the algorithm discovers structure on its own.
```

**Visual intuition:**
```
Before K-Means (raw data):           After K-Means (K=3):
  ●  ●   ●                             ■  ■   ■          Cluster 1 (■)
●   ● ●  ●                           ■   ■ ■  ■
         ▲ ▲ ▲                                ▲ ▲ ▲        Cluster 2 (▲)
       ▲   ▲                               ▲   ▲
 ◆  ◆     ◆                           ◆  ◆     ◆          Cluster 3 (◆)
◆     ◆  ◆                           ◆     ◆  ◆
```

> 💡 "K-Means answers: 'Which group does this point naturally belong to?'  
>      without any labels — purely from geometric closeness."

---

## 🔍 When to Use K-Means?

| Condition | Use K-Means? |
|-----------|:-----------:|
| Discovering natural groups in data | ✅ Yes |
| Clusters are roughly spherical | ✅ Yes |
| Need fast, scalable clustering | ✅ Yes |
| Preprocessing for supervised learning | ✅ Yes |
| Customer segmentation, image compression | ✅ Yes |
| Clusters have very different sizes/shapes | ❌ No → DBSCAN / GMM |
| Clusters are non-convex (ring, crescent) | ❌ No → DBSCAN |
| Number of clusters unknown | ⚠️ Need elbow/silhouette methods |
| Data has many outliers | ⚠️ Caution — K-Means is sensitive to outliers |

---

## 🧮 The Algorithm — Lloyd's Algorithm

```
Step 1: Initialize K centroids
        Method: 'k-means++' (smart initialization) or random

Step 2: Assignment step
        For each point xᵢ:
          Assign to cluster k* where k* = argmin_k ||xᵢ − μₖ||²
          (nearest centroid by Euclidean distance)

Step 3: Update step
        For each cluster k:
          μₖ = (1/nₖ) × Σ xᵢ   (mean of all assigned points)
                         i∈k

Step 4: Repeat Steps 2–3 until:
        - Centroids stop moving (convergence)
        - OR max_iter reached

Output: Cluster labels {0, 1, ..., K-1} for each point
        Centroids μ₁, μ₂, ..., μₖ
```

---

## 📐 Objective Function — Inertia (WCSS)

K-Means minimizes the **Within-Cluster Sum of Squares (WCSS)**, also called inertia:

```
Inertia = Σₖ Σᵢ∈k ||xᵢ − μₖ||²

Where:
  K   = number of clusters
  μₖ  = centroid of cluster k
  xᵢ  = data point in cluster k

Properties:
  → Lower inertia = more compact, tighter clusters
  → Inertia always decreases as K increases (trivially 0 when K=n)
  → Finding global minimum is NP-hard — K-Means finds local minimum
  → Multiple random initializations help escape poor local minima
```

---

## 🎯 K-Means++ Initialization

Standard random initialization can lead to poor convergence.  
**K-Means++** chooses initial centroids that are spread far apart:

```
Algorithm:
  1. Pick first centroid uniformly at random from data points
  2. For each remaining centroid:
     a. Compute D(x) = distance from each point to nearest existing centroid
     b. Pick next centroid with probability ∝ D(x)²
        (farther points more likely to be chosen)
  3. Repeat until K centroids selected

Benefits:
  ✅ Better initial centroids → faster convergence
  ✅ More likely to find global optimum
  ✅ O(log K) approximation to optimal solution

sklearn default: init='k-means++'
```

---

## 🔢 Choosing K — The Right Number of Clusters

### Method 1: Elbow Method

```
Plot inertia (WCSS) vs K:

Inertia
│●
│ ●
│  ●
│   ●                 ← Elbow: diminishing returns after this point
│     ●●
│        ●●●●●●●●●●
└──────────────────── K
  1  2  3  4  5  6  7

Choose K at the "elbow" — where adding another cluster
gives diminishing improvement in inertia.

⚠️ Elbow is not always clearly visible.
```

### Method 2: Silhouette Score

```
For each point i, silhouette score s(i):

  s(i) = (b(i) − a(i)) / max(a(i), b(i))

Where:
  a(i) = mean distance to all OTHER points in the SAME cluster
          (intra-cluster distance — how well it fits its own cluster)
  b(i) = mean distance to all points in the NEAREST OTHER cluster
          (inter-cluster distance — how far from the next cluster)

Range:
  s(i) = +1 → well inside its cluster (ideal)
  s(i) =  0 → on the boundary between clusters
  s(i) = -1 → likely in the wrong cluster

Overall score = mean of all s(i)
Choose K with the HIGHEST silhouette score

sklearn: from sklearn.metrics import silhouette_score
```

### Method 3: Gap Statistic

```
Compares inertia to a reference random distribution:
  Gap(K) = E[log(W_ref(K))] − log(W(K))

Choose K where Gap(K) is maximized.
More statistically rigorous but computationally expensive.
```

### Method 4: Calinski-Harabasz Index

```
Ratio of between-cluster dispersion to within-cluster dispersion.
Higher = better defined clusters.

sklearn: from sklearn.metrics import calinski_harabasz_score
```

---

## ⚖️ Feature Scaling — Critical for K-Means

K-Means uses Euclidean distance — features with large ranges dominate:

```
Without scaling:
  Age: [0–100]         → dominates distance calculation
  Income: [0–200,000]  → completely dominates!
  Score: [0–1]         → nearly ignored

With StandardScaler:
  All features → mean=0, std=1
  → Equal contribution to distance ✅

Always apply StandardScaler before K-Means.
```

---

## ⚠️ Limitations of K-Means

### 1. Assumes Spherical Clusters
```
K-Means works well for:     K-Means fails for:
  ●●●    ●●●                 (ring shapes)
 ●●●●●  ●●●●●                (crescent shapes)
  ●●●    ●●●                 (elongated clusters)
(roughly circular)           → Use DBSCAN or GMM
```

### 2. Sensitive to Outliers
```
Centroid = mean of all points in cluster
→ A single extreme outlier can pull the centroid far from the true center

Fix: Use K-Medoids (uses actual data points as centers)
     Or: Remove outliers before clustering
```

### 3. Requires K in Advance
```
Must specify K before training
→ Use elbow method / silhouette analysis to estimate K
```

### 4. Local Minima
```
K-Means can get stuck in poor local solutions
Fix: Run multiple times with different initializations
     sklearn: n_init=10 (default) — runs 10 times, keeps best
```

---

## 🔄 Mini-Batch K-Means

For very large datasets, use **Mini-Batch K-Means** — processes small random  
batches of data instead of the full dataset per iteration:

```python
from sklearn.cluster import MiniBatchKMeans

model = MiniBatchKMeans(n_clusters=K, batch_size=1000, random_state=42)
```

```
Trade-off:
  ✅ Much faster (O(batch_size) per iteration vs O(n))
  ✅ Lower memory usage
  ⚠️ Slightly worse clustering quality than full K-Means
  Best for: n > 100,000 rows
```

---

## 🎛️ Key Hyperparameters

| Parameter | Effect | Typical Values |
|-----------|--------|----------------|
| `n_clusters` (K) | Number of clusters | Determined by elbow/silhouette |
| `init` | Initialization method | 'k-means++' (default), 'random' |
| `n_init` | Number of random restarts | 10 (default), increase for stability |
| `max_iter` | Max iterations per run | 300 (default) |
| `tol` | Convergence tolerance | 1e-4 (default) |
| `algorithm` | Computation method | 'lloyd' (default), 'elkan' |

---

## 📊 Evaluation Metrics (Unsupervised)

| Metric | Formula | Notes |
|--------|---------|-------|
| **Inertia (WCSS)** | Σ||xᵢ − μₖ||² | Lower = more compact (scale-dependent) |
| **Silhouette Score** | (b−a)/max(a,b) | Range [−1,1] — higher = better |
| **Calinski-Harabasz** | between/within dispersion | Higher = better |
| **Davies-Bouldin** | avg max cluster similarity | Lower = better |

**If labels available** (semi-supervised evaluation):

| Metric | Notes |
|--------|-------|
| **Adjusted Rand Index (ARI)** | Range [−1,1] — 1 = perfect agreement with true labels |
| **Normalized Mutual Information (NMI)** | Range [0,1] — information between clusters and labels |
| **Homogeneity / Completeness** | Each cluster contains one class / all members in one cluster |

---

## 🆚 K-Means vs Other Clustering Methods

| Aspect | K-Means | DBSCAN | GMM | Hierarchical |
|--------|:-------:|:------:|:---:|:------------:|
| Need to specify K | ✅ Yes | ❌ No | ✅ Yes | ❌ No (dendrogram) |
| Cluster shape | Spherical only | Any shape | Elliptical | Any |
| Outlier handling | ❌ Sensitive | ✅ Labels outliers | ⚠️ Medium | ❌ Sensitive |
| Scalability | ✅ Fast (O(nKI)) | ⚠️ O(n log n) | ❌ Slow | ❌ O(n²) |
| Probabilistic | ❌ Hard assignment | ❌ Hard | ✅ Soft | ❌ Hard |
| Deterministic | ⚠️ Local minima | ✅ Yes | ⚠️ Local | ✅ Yes |

---

## 🔗 Related Topics

- `04_Unsupervised_Learning/DBSCAN` — Density-based, handles non-convex clusters
- `04_Unsupervised_Learning/Hierarchical` — No K needed, builds dendrogram
- `04_Unsupervised_Learning/PCA` — Dimensionality reduction before clustering
- `05_Model_Evaluation` — Silhouette score, ARI for cluster evaluation
- `K_Nearest_Neighbors` — Also distance-based, but supervised

---

## 📚 References

- Scikit-learn `KMeans`: [https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- Scikit-learn Clustering Guide: [https://scikit-learn.org/stable/modules/clustering.html](https://scikit-learn.org/stable/modules/clustering.html)
- K-Means++ Paper (Arthur & Vassilvitskii, 2007)
- An Introduction to Statistical Learning — Chapter 12.4
- The Elements of Statistical Learning — Chapter 14.3
