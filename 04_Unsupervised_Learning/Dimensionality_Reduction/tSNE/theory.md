# 📘 t-SNE — Theory

---

## 📌 What is t-SNE?

t-SNE (**t-distributed Stochastic Neighbor Embedding**) is a **non-linear  
dimensionality reduction technique** primarily designed for **visualizing  
high-dimensional data** in 2D or 3D. It preserves the **local structure**  
(nearby points stay nearby) rather than global structure.

```
High-dimensional space (d dimensions):   Low-dimensional space (2D):
  x₁, x₂, ..., xd         →  t-SNE  →     y₁, y₂

  Complex, overlapping         Compact, separated clusters
  clusters in many dims        that reveal natural groupings
```

**Visual intuition:**
```
Before t-SNE (raw high-d data):    After t-SNE (2D projection):
  ● ◆ ▲ ● ◆ ▲ ● ◆ ▲               ●●●●●
  ◆ ● ▲ ◆ ● ▲ ◆ ▲ ●               (cluster 1)
  (all mixed up)                        ◆◆◆◆◆
                                        (cluster 2)
                                              ▲▲▲▲▲
                                              (cluster 3)
```

> 💡 "t-SNE is a microscope for high-dimensional data —  
>      it reveals hidden cluster structure that is invisible  
>      in the raw feature space."

---

## 🔍 When to Use t-SNE?

| Condition | Use t-SNE? |
|-----------|:---------:|
| Visualize high-dimensional clusters | ✅ Yes — primary use |
| Explore class separation before modeling | ✅ Yes |
| Understand embedding quality | ✅ Yes |
| Need to project new data | ❌ No — no transform() method |
| Need interpretable components | ❌ No — axes have no meaning |
| Need to preserve global structure | ❌ No → PCA or UMAP |
| Use as input features for ML | ❌ No — non-reproducible axes |
| Large dataset (> 100K rows) | ❌ No → O(n²), too slow; use UMAP |

---

## 🧮 The Algorithm — Step by Step

### Step 1: Compute Pairwise Similarities in High-D Space

```
For each pair of points (i, j) in the original d-dimensional space:
  Compute the conditional probability that point i would pick j as neighbor:

            exp(−||xᵢ − xⱼ||² / 2σᵢ²)
  p(j|i) = ─────────────────────────────────
              Σₖ≠ᵢ exp(−||xᵢ − xₖ||² / 2σᵢ²)

  Where σᵢ is the bandwidth of the Gaussian centered at xᵢ
  (chosen via perplexity — see below)

Symmetrize:
  pᵢⱼ = (p(j|i) + p(i|j)) / 2n

Result: pᵢⱼ ≈ HIGH if xᵢ and xⱼ are close
         pᵢⱼ ≈ LOW  if xᵢ and xⱼ are far
```

### Step 2: Define Similarities in Low-D Space (t-distribution)

```
For points yᵢ, yⱼ in the 2D embedding:
  Use Student's t-distribution (heavy-tailed) instead of Gaussian:

           (1 + ||yᵢ − yⱼ||²)⁻¹
  qᵢⱼ = ──────────────────────────────────
          Σₖ≠ₗ (1 + ||yₖ − yₗ||²)⁻¹

Why t-distribution?
  → Heavy tails allow moderately distant points to be placed MUCH farther apart
  → Prevents crowding of points in the low-d space (the "crowding problem")
  → Creates clear visual separation between clusters
```

### Step 3: Minimize KL Divergence

```
Objective: make qᵢⱼ match pᵢⱼ as closely as possible

Loss = KL(P || Q) = Σᵢ Σⱼ pᵢⱼ × log(pᵢⱼ / qᵢⱼ)

Optimization: Gradient descent in the 2D embedding space

∂KL/∂yᵢ = 4 × Σⱼ (pᵢⱼ − qᵢⱼ)(yᵢ − yⱼ)(1 + ||yᵢ − yⱼ||²)⁻¹

→ Points with high pᵢⱼ but low qᵢⱼ: attract each other in 2D
→ Points with low pᵢⱼ but high qᵢⱼ: repel each other in 2D
```

---

## 🎛️ Key Hyperparameters

### 1. Perplexity

```
Perplexity ≈ effective number of neighbors considered for each point

Low perplexity (5–10):
  → Very local focus — only closest neighbors matter
  → Clusters may fragment into many small sub-clusters
  → Reveals micro-structure

High perplexity (30–50):
  → Broader neighborhood → more global-ish view
  → Smoother, larger clusters
  → Better for large, dense datasets

Typical range: 5–50
Rule of thumb: perplexity ≈ √n  (where n = dataset size)
sklearn default: 30

⚠️ ALWAYS run t-SNE with multiple perplexity values
    Different values reveal different aspects of structure
```

### 2. Learning Rate (eta)

```
Controls step size during gradient descent optimization.

Too small  → slow convergence, points cluster in center
Too large  → points explode to edges, no structure visible
Typical    : 100–1000
sklearn default: 'auto' = max(n/early_exaggeration, 50)

Note: sklearn >= 1.2 uses 'auto' by default (recommended)
```

### 3. n_iter (Number of Iterations)

```
Optimization runs in two phases:
  Phase 1 (early exaggeration): pᵢⱼ artificially inflated
                                 → forces natural cluster formation
  Phase 2 (refinement):        normal optimization
                                 → fine-tunes cluster positions

n_iter < 250  → usually insufficient (too few iterations)
n_iter = 1000 → sklearn default (usually enough)
n_iter = 5000 → better convergence for complex data
```

### 4. Early Exaggeration

```
Multiplier applied to pᵢⱼ in early optimization phase:
  Large pᵢⱼ → tight initial clustering → clusters form more distinctly

sklearn default: 12.0 (usually no need to change)
```

---

## ⚠️ Critical Pitfalls and Misinterpretations

### 1. Distances Between Clusters ARE NOT Meaningful
```
In t-SNE output:
  ✅ Points that are close = similar in original space
  ❌ Distance between clusters ≠ actual similarity between clusters
  ❌ Cluster A being "closer" to B than C is NOT meaningful
```

### 2. Cluster Sizes ARE NOT Meaningful
```
In t-SNE:
  ❌ A large cluster is NOT necessarily a larger/denser group
  ❌ A small cluster is NOT necessarily a rare group
  → Cluster sizes depend on perplexity and local density, not actual sizes
```

### 3. Different Runs Give Different Layouts
```
t-SNE uses random initialization:
  → Two runs with same data give topologically similar but
     geometrically different plots (rotated, reflected, rearranged)
  → Set random_state for reproducibility
  → Clusters will be the same, but their positions may differ
```

### 4. Perplexity Too Low → Fragmented Clusters
```
perplexity=5 on data with 500 points per class:
  → Each cluster fragments into 10–20 sub-clusters
  → Looks like many small groups instead of a few large ones
  → NOT real sub-structure (just an artifact of perplexity)
```

### 5. Cannot Transform New Points
```
t-SNE has no transform() method:
  → Must re-run the entire algorithm with new data included
  → Cannot add a new point to an existing t-SNE embedding
  → For production: use PCA or UMAP instead
```

---

## 📊 Perplexity Effect — Visual Summary

```
Same dataset, different perplexity:

Perplexity = 5:              Perplexity = 30:          Perplexity = 100:
  ● ●● ● ● ●                    ●●●●●                    ●●●●●●●
   ● ●●●● ●                    ●●●●●●●                  ●●●●●●●●●
  ● ● ●● ●                    ●●●●●●●●                 ●●●●●●●●●●

  (fragmented —               (good clusters —          (over-smoothed —
   too many small              natural structure          losing local detail)
   sub-clusters)               visible)
```

---

## 🔄 t-SNE vs PCA vs UMAP

| Aspect | t-SNE | PCA | UMAP |
|--------|:-----:|:---:|:----:|
| Linear | ❌ No | ✅ Yes | ❌ No |
| Preserves local structure | ✅ Excellent | ⚠️ Partial | ✅ Excellent |
| Preserves global structure | ❌ No | ✅ Yes | ⚠️ Partial |
| Speed | ❌ Slow O(n²) | ✅ Fast | ✅ Fast O(n log n) |
| Can transform new data | ❌ No | ✅ Yes | ✅ Yes |
| Deterministic | ❌ No (random) | ✅ Yes | ❌ No (random) |
| Best for | Visualization | Preprocessing | Both |
| Interpretable axes | ❌ No | ✅ Yes | ❌ No |
| Scalable (large n) | ❌ No | ✅ Yes | ✅ Yes |

---

## ⚡ Barnes-Hut t-SNE (Approximate, Faster)

```
Standard t-SNE: O(n²) — infeasible for n > 10,000
Barnes-Hut t-SNE: O(n log n) — handles up to ~100,000 points

Key idea:
  Approximate distant repulsive forces using tree structure
  → Nearby interactions: exact
  → Distant interactions: approximated via space-partitioning tree

sklearn: TSNE(method='barnes_hut', angle=0.5)  ← default for large n
         TSNE(method='exact')                   ← for small n (< 5000)

angle: trade-off between speed and accuracy
  0.2 → more accurate, slower
  0.8 → faster, less accurate
  0.5 → sklearn default
```

---

## 🎛️ sklearn Implementation

```python
from sklearn.manifold import TSNE

# Standard usage
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate='auto',
    n_iter=1000,
    random_state=42,
    init='pca',        # better than random init
    method='barnes_hut',
    n_jobs=-1,
)
X_tsne = tsne.fit_transform(X_scaled)

# For large datasets: reduce with PCA first
from sklearn.decomposition import PCA
X_pca_50  = PCA(n_components=50).fit_transform(X_scaled)
X_tsne_2d = TSNE(n_components=2, perplexity=30,
                  random_state=42).fit_transform(X_pca_50)
```

---

## 🔗 Related Topics

- `PCA` — Fast linear alternative; use PCA first to reduce to 50D before t-SNE
- `LDA` — Supervised reduction; useful for class-labeled visualization
- `04_Unsupervised_Learning/KMeans` — Apply after t-SNE for cluster analysis
- `04_Unsupervised_Learning/DBSCAN` — Combine with t-SNE 2D output for clustering

---

## 📚 References

- Scikit-learn `TSNE`: [https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
- Original t-SNE Paper (van der Maaten & Hinton, 2008): [https://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf](https://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
- "How to Use t-SNE Effectively" (Wattenberg et al., 2016): [https://distill.pub/2016/misread-tsne/](https://distill.pub/2016/misread-tsne/)
- Barnes-Hut t-SNE (van der Maaten, 2014): [https://arxiv.org/abs/1301.3342](https://arxiv.org/abs/1301.3342)
