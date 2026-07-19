# 📘 Hierarchical Clustering — Theory

---

## 📌 What is Hierarchical Clustering?

Hierarchical Clustering builds a **tree-like structure (dendrogram)** of clusters  
by successively merging (agglomerative) or splitting (divisive) data points —  
without requiring the number of clusters K in advance.

```
Agglomerative (bottom-up) — most common:
  Start: each point is its own cluster
  Step:  merge the two closest clusters
  End:   all points in one cluster

Divisive (top-down) — less common:
  Start: all points in one cluster
  Step:  split the least cohesive cluster
  End:   each point is its own cluster
```

**Dendrogram — the output:**
```
Height
  │
8 │                     ┌────────────────┐
  │                     │                │
6 │            ┌────────┤         ┌──────┤
  │            │        │         │      │
4 │      ┌─────┤    ┌───┤    ┌────┤  ┌───┤
  │      │     │    │   │    │    │  │   │
  └──────┴─────┴────┴───┴────┴────┴──┴───┴
         A     B    C   D    E    F  G   H
                     Data Points

Cut at height 4 → 3 clusters: {A,B}, {C,D}, {E,F,G,H}
Cut at height 6 → 2 clusters: {A,B,C,D}, {E,F,G,H}
```

> 💡 "Hierarchical Clustering gives you a full picture of all possible groupings  
>      at once — you choose the cut level after seeing the dendrogram."

---

## 🔍 When to Use Hierarchical Clustering?

| Condition | Use Hierarchical Clustering? |
|-----------|:---------------------------:|
| Number of clusters unknown | ✅ Yes — key advantage |
| Want to visualize cluster hierarchy | ✅ Yes — dendrogram |
| Small to medium dataset (< 10K rows) | ✅ Yes |
| Need interpretable, reproducible result | ✅ Yes — deterministic |
| Non-spherical cluster shapes | ✅ Yes (with correct linkage) |
| Very large dataset (> 100K rows) | ❌ No → O(n²) memory |
| Need fast clustering | ❌ No → K-Means is faster |

---

## 🧮 The Agglomerative Algorithm

```
Input:  n data points, linkage criterion, distance metric
Output: Dendrogram + cluster labels at chosen cut level

Step 1: Compute pairwise distance matrix (n × n)
        D[i,j] = distance between points i and j

Step 2: Initialize: each point = its own cluster
        Clusters = {x₁}, {x₂}, ..., {xₙ}

Step 3: Find the two closest clusters Cᵢ, Cⱼ:
        (Cᵢ, Cⱼ) = argmin D(Cᵢ, Cⱼ)

Step 4: Merge Cᵢ and Cⱼ into one cluster

Step 5: Update distance matrix:
        Compute distance from new cluster to all others
        (method depends on linkage criterion)

Step 6: Repeat Steps 3–5 until one cluster remains

Step 7: Cut the dendrogram at desired height → get K clusters
```

**Complexity:**
```
Time  : O(n² log n) with efficient implementations
Memory: O(n²) — stores full pairwise distance matrix
→ Infeasible for n > ~10,000 rows
```

---

## 🔗 Linkage Criteria — How to Measure Cluster Distance

The **linkage criterion** defines the distance between two clusters.  
This is the most important choice in hierarchical clustering:

---

### 1. Single Linkage (Minimum)
```
D(A, B) = min { d(a, b) : a ∈ A, b ∈ B }
           (distance between closest pair of points)

Properties:
  ✅ Can find non-convex, elongated clusters
  ✅ Sensitive to the "path" between clusters
  ❌ Chaining effect — clusters can form long chains
  ❌ Sensitive to outliers and noise
  Best for: Elongated or irregular shapes
```

### 2. Complete Linkage (Maximum)
```
D(A, B) = max { d(a, b) : a ∈ A, b ∈ B }
           (distance between farthest pair of points)

Properties:
  ✅ Produces compact, roughly spherical clusters
  ✅ Less sensitive to outliers than single linkage
  ❌ Can break large clusters to satisfy the max distance criterion
  Best for: Compact, spherical clusters
```

### 3. Average Linkage (UPGMA)
```
D(A, B) = (1/|A||B|) × Σₐ∈A Σᵦ∈B d(a, b)
            (average distance between all pairs)

Properties:
  ✅ Compromise between single and complete
  ✅ Less prone to chaining and compactness bias
  ✅ Generally robust
  Best for: General purpose (often best default)
```

### 4. Ward Linkage (Minimum Variance)
```
D(A, B) = increase in total within-cluster variance
           when A and B are merged

ΔE(A, B) = (|A|×|B|)/(|A|+|B|) × ||μ_A − μ_B||²

Properties:
  ✅ Minimizes total within-cluster variance → compact clusters
  ✅ Tends to produce equally-sized clusters
  ✅ Generally produces the best results on most datasets
  ❌ Only works with Euclidean distance
  Best for: General purpose (sklearn default)

sklearn default: linkage='ward', metric='euclidean'
```

### 5. Centroid Linkage
```
D(A, B) = d(μ_A, μ_B)
           (distance between cluster centroids)

Properties:
  ⚠️ Can cause inversions in the dendrogram
  Generally not recommended
```

---

## 📏 Distance Metrics

The distance metric measures similarity between individual points:

```
Euclidean (L2) : √(Σ(xᵢ − yᵢ)²)     → default, good for continuous data
Manhattan (L1) : Σ|xᵢ − yᵢ|          → robust to outliers
Cosine          : 1 − cos(angle)       → good for text, directional data
Correlation     : 1 − Pearson r        → shape-based similarity
Chebyshev       : max|xᵢ − yᵢ|        → max dimension difference

⚠️ Ward linkage only works with Euclidean distance.
   For other metrics, use average or complete linkage.
```

---

## 🌳 Reading the Dendrogram

```
     Height
       │
  12   │                    ┌──────────────────────┐
       │                    │                      │
   8   │         ┌──────────┤              ┌───────┤
       │         │          │              │       │
   5   │   ┌─────┤    ┌─────┤       ┌─────┤  ┌────┤
       │   │     │    │     │       │     │  │    │
   2   │ ┌─┤  ┌──┤  ┌─┤  ┌──┤    ┌──┤  ┌─┤  │  ┌─┤
       └──┴─┴──┴──┴──┴─┴──┴──┴────┴──┴──┴─┴──┴──┴─┴
            A  B  C  D  E  F        G  H  I  J  K

Interpretation:
  Height of merge = dissimilarity between merged clusters
  Long vertical lines → clusters were far apart (natural separation)
  Short vertical lines → clusters were close (less meaningful merge)

Choosing cut height:
  Cut where the vertical lines are LONGEST (biggest jumps in height)
  → These represent the most natural cluster boundaries
  → The number of horizontal lines crossed = number of clusters

Color threshold in scipy:
  dendrogram(Z, color_threshold=8) → cuts at height 8 → 3 clusters
```

---

## ✂️ Cutting the Dendrogram

```python
from scipy.cluster.hierarchy import fcluster

# Method 1: Cut at a specific height
labels = fcluster(Z, t=8.0, criterion='distance')

# Method 2: Specify number of clusters directly
labels = fcluster(Z, t=3, criterion='maxclust')

# Method 3: Inconsistency coefficient
labels = fcluster(Z, t=1.5, criterion='inconsistent')
```

---

## 🔢 Choosing the Number of Clusters

### From the Dendrogram
```
Look for the largest vertical gap (longest vertical line) before cutting:

  ─────────────── ← Cut here (longest gap below = most natural K)
       gap 4       → K = 3 clusters
  ─────────────── ← This would be K = 2
       gap 2
  ─────────────── ← K = 4
       gap 1
```

### Using Silhouette Score
```
Fit with different K values (cut dendrogram at different heights)
Choose K with highest silhouette score (same as K-Means selection)
```

---

## 🎛️ sklearn API

```python
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(
    n_clusters=3,          # Number of clusters (or None for full tree)
    linkage='ward',        # 'ward', 'complete', 'average', 'single'
    metric='euclidean',    # 'euclidean', 'manhattan', 'cosine', etc.
    compute_full_tree=True # Needed if n_clusters=None
)
labels = model.fit_predict(X_scaled)

# For dendrogram: use scipy
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
Z = linkage(X_scaled, method='ward')  # linkage matrix
```

---

## 🆚 Agglomerative vs K-Means vs DBSCAN

| Aspect | Hierarchical | K-Means | DBSCAN |
|--------|:------------:|:-------:|:------:|
| Need K in advance | ❌ No | ✅ Yes | ❌ No |
| Deterministic | ✅ Yes | ❌ No | ✅ Yes |
| Non-spherical shapes | ✅ (with single/avg) | ❌ | ✅ |
| Handles outliers | ⚠️ Sensitive | ⚠️ Sensitive | ✅ Labels as noise |
| Scalability | ❌ O(n²) | ✅ O(nKI) | ✅ O(n log n) |
| Visualization | ✅ Dendrogram | ❌ | ❌ |
| Reproducibility | ✅ Yes | ⚠️ Local minima | ✅ Yes |

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Not scaling features | Euclidean distance biased by large-range features | Always StandardScale first |
| Using Ward with non-Euclidean metric | Ward requires Euclidean | Use average/complete with other metrics |
| Ignoring dendrogram before cutting | Miss natural cluster structure | Always plot dendrogram first |
| Large dataset (n > 10K) | O(n²) memory / time infeasible | Use K-Means or mini-batch variants |
| Chaining with single linkage | Long chain clusters, poor quality | Use Ward or average linkage instead |
| Not validating with silhouette | No quality check | Always compute silhouette after cutting |

---

## 🔗 Related Topics

- `KMeans` — Faster alternative, requires K in advance
- `DBSCAN` — Density-based, finds arbitrary shapes, labels outliers
- `04_Unsupervised_Learning/PCA` — Reduce dimensions before clustering
- `05_Model_Evaluation` — Silhouette, ARI for cluster evaluation

---

## 📚 References

- Scikit-learn `AgglomerativeClustering`: [https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
- Scipy `linkage` / `dendrogram`: [https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)
- An Introduction to Statistical Learning — Chapter 12.4
- The Elements of Statistical Learning — Chapter 14.3
