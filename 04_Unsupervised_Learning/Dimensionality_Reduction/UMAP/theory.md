# 📘 UMAP — Theory

---

## 📌 What is UMAP?

UMAP (**Uniform Manifold Approximation and Projection**) is a **non-linear  
dimensionality reduction technique** that preserves both **local and global  
structure** of high-dimensional data — and projects it into a low-dimensional  
space (typically 2D or 3D) for visualization or downstream ML tasks.

```
High-dimensional space (d dimensions):       Low-dimensional space (2D):
  x₁, x₂, ..., xd         →  UMAP  →          y₁, y₂

  Complex manifold structure                   Faithful 2D layout
  Clusters, gradients, topology                Local + global preserved
```

**Visual intuition:**
```
t-SNE output:               UMAP output:
  [●●●]  [◆◆◆]               [●●●]──[◆◆◆]
      [▲▲▲]                       [▲▲▲]

  Clusters clear             Clusters clear +
  Distances NOT meaningful   relative distances MORE meaningful
```

> 💡 "UMAP is like t-SNE with a GPS — it shows you where clusters are  
>      AND gives you a better sense of how far apart they really are."

---

## 🔍 When to Use UMAP?

| Condition | Use UMAP? |
|-----------|:---------:|
| Visualize high-dimensional data in 2D/3D | ✅ Yes — primary strength |
| Clusters have complex non-linear structure | ✅ Yes |
| Need to project new points (transform) | ✅ Yes — unlike t-SNE |
| Large dataset (> 10K rows) | ✅ Yes — O(n log n) vs t-SNE O(n²) |
| Need reproducible results | ✅ Yes (with `random_state`) |
| Need interpretable axes | ❌ No — axes have no meaning |
| Global distances must be perfectly preserved | ❌ No → PCA |
| Use as input features for ML | ⚠️ Possible but caution — non-reproducible |

---

## 🧮 Mathematical Foundations

UMAP is grounded in **topological data analysis** and **Riemannian geometry**,  
but the key intuition is:

### Step 1: Build a Fuzzy Topological Representation (High-D)

```
For each point xᵢ, find its k nearest neighbors.

Compute a local connectivity metric using an adaptive distance:
  Distance is normalized by the distance to the nearest neighbor (ρᵢ)
  and a local bandwidth (σᵢ)

Membership strength of edge (i, j):
  w(i,j) = exp(−max(0, d(xᵢ, xⱼ) − ρᵢ) / σᵢ)

This creates a "fuzzy simplicial set" — a weighted graph where:
  Nearby neighbors → high edge weight (close to 1)
  Far neighbors    → low edge weight (close to 0)

σᵢ is chosen so each point has exactly log₂(n_neighbors) effective connections
```

### Step 2: Build a Low-D Representation

```
Initialize 2D positions (using spectral embedding or random)

Define low-D similarity using a modified Cauchy distribution:
  v(i,j) = (1 + a × ||yᵢ − yⱼ||^(2b))⁻¹

Where a, b are hyperparameters determined by min_dist
  min_dist small → tight, compact clusters
  min_dist large → spread-out, diffuse layout
```

### Step 3: Optimize via Cross-Entropy Loss

```
Minimize the fuzzy cross-entropy between high-D and low-D graphs:

L = Σᵢⱼ [w(i,j) × log(w(i,j)/v(i,j)) + (1−w(i,j)) × log((1−w(i,j))/(1−v(i,j)))]

Optimization: Stochastic Gradient Descent with negative sampling
  → Attractive forces: pull together high-similarity pairs
  → Repulsive forces: push apart low-similarity pairs
```

---

## 🆚 UMAP vs t-SNE — Key Differences

| Aspect | UMAP | t-SNE |
|--------|:----:|:-----:|
| Speed | ✅ Fast O(n log n) | ❌ Slow O(n²) |
| Global structure | ✅ Better preserved | ❌ Poor |
| Local structure | ✅ Excellent | ✅ Excellent |
| Can transform new data | ✅ Yes (`transform()`) | ❌ No |
| Deterministic | ✅ Yes (with seed) | ❌ No (random) |
| Scalability | ✅ Large datasets | ❌ < ~50K rows |
| Theoretical foundation | Topology/Riemannian | KL divergence |
| Cluster distances meaningful | ⚠️ More than t-SNE | ❌ Not at all |
| Hyperparameter sensitivity | ⚠️ Moderate | ⚠️ High |

---

## 🎛️ Key Hyperparameters

### 1. n_neighbors

```
Controls how many nearest neighbors define each point's local structure.

Low n_neighbors (2–5):
  → Very local focus — captures fine-grained micro-structure
  → Can fragment clusters into many small sub-clusters
  → Faster computation

High n_neighbors (50–200):
  → Broader neighborhood → more global view
  → Smoother, larger clusters
  → Slower computation

Typical range: 5–50
Default: 15

Rule of thumb:
  Small dataset  → 5–15
  Large dataset  → 15–50
```

### 2. min_dist

```
Controls how tightly points are packed in the low-dimensional space.

Low min_dist (0.0–0.1):
  → Points within clusters packed very tightly
  → Clusters appear compact and dense
  → Good for seeing cluster membership

High min_dist (0.5–1.0):
  → Points spread out within clusters
  → More even distribution of points
  → Good for seeing continuous structure / gradients

Default: 0.1
Typical range: 0.0–0.5
```

### 3. n_components

```
Number of output dimensions:
  2 → 2D scatter plot (most common)
  3 → 3D visualization
  k → General dimensionality reduction for ML pipelines

For visualization: always 2 or 3
For ML preprocessing: typically 5–50
```

### 4. metric

```
Distance metric for computing high-dimensional similarity:
  'euclidean'  → default, good for continuous features
  'manhattan'  → robust to outliers
  'cosine'     → good for text/NLP features
  'correlation'→ shape-based similarity
  'hamming'    → binary/categorical data
  'jaccard'    → set-based similarity

Always StandardScale before UMAP with Euclidean metric.
```

---

## 📦 Installation and Import

```python
# Install
pip install umap-learn

# Import
import umap

# Usage
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric='euclidean',
    random_state=42,
    n_jobs=-1,
)
X_umap = reducer.fit_transform(X_scaled)

# Transform new data (unlike t-SNE!)
X_new_umap = reducer.transform(X_new_scaled)
```

---

## 🔄 UMAP for Dimensionality Reduction (ML Preprocessing)

Unlike t-SNE, UMAP can be used as a **preprocessing step** for ML:

```python
# Use as sklearn-compatible transformer
from umap import UMAP
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('umap',   UMAP(n_components=10, random_state=42)),
    ('model',  RandomForestClassifier()),
])
pipe.fit(X_train, y_train)

# Can transform new data → production-ready ✅
```

```
When to use UMAP for ML (not just visualization):
  ✅ High-dimensional data (images, text embeddings)
  ✅ When PCA misses non-linear structure
  ✅ When downstream model benefits from manifold structure
  ⚠️ Not recommended when interpretability needed
  ⚠️ Be careful: UMAP is semi-stochastic even with random_state
```

---

## 📊 Parametric UMAP (Neural Network-Based)

Standard UMAP learns a fixed embedding — no parametric function.  
**Parametric UMAP** trains a neural network to learn the mapping:

```
Benefits:
  ✅ True out-of-sample generalization
  ✅ Faster inference for new points
  ✅ Can be fine-tuned on new data

Requires: TensorFlow
pip install umap-learn[parametric_umap]

from umap.parametric_umap import ParametricUMAP
reducer = ParametricUMAP()
```

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Not scaling before UMAP | Distance-based, dominated by large-range features | Always StandardScale first |
| Taking cluster distances literally | More meaningful than t-SNE but still not Euclidean distances | Treat as qualitative, not quantitative |
| Using single n_neighbors value | Different values reveal different structure | Try multiple: 5, 15, 50 |
| Expecting perfect reproducibility | UMAP has stochastic elements even with seed | Set `random_state`, accept minor variation |
| High-d data without PCA pre-reduction | Slow and noisy | Apply PCA to 50D first, then UMAP |
| Using UMAP axes as features carelessly | Non-linear mapping → can cause leakage | Only use UMAP features in careful pipeline |

---

## 🔗 Related Topics

- `tSNE` — Non-linear alternative; compare outputs side by side
- `PCA` — Linear alternative; use PCA→UMAP for large high-d data
- `04_Unsupervised_Learning/KMeans` — Apply KMeans on UMAP 2D output
- `04_Unsupervised_Learning/DBSCAN` — Cluster the UMAP embedding

---

## 📚 References

- UMAP Documentation: [https://umap-learn.readthedocs.io/](https://umap-learn.readthedocs.io/)
- Original UMAP Paper (McInnes et al., 2018): [https://arxiv.org/abs/1802.03426](https://arxiv.org/abs/1802.03426)
- How to Use UMAP (official guide): [https://umap-learn.readthedocs.io/en/latest/basic_usage.html](https://umap-learn.readthedocs.io/en/latest/basic_usage.html)
- Parametric UMAP: [https://umap-learn.readthedocs.io/en/latest/parametric_umap.html](https://umap-learn.readthedocs.io/en/latest/parametric_umap.html)
