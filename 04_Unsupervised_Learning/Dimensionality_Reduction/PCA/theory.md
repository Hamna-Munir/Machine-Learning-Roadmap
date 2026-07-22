# 📘 Principal Component Analysis (PCA) — Theory

---

## 📌 What is PCA?

Principal Component Analysis is an **unsupervised dimensionality reduction technique**  
that transforms a dataset of correlated features into a smaller set of **uncorrelated  
components** (principal components) that capture the **maximum variance** in the data.

```
Original space (d dimensions):        Reduced space (k dimensions, k << d)
  x₁, x₂, x₃, ..., xd     →PCA→     PC1, PC2, ..., PCk

  Correlated, redundant features         Uncorrelated, ordered by variance
  All equal importance                   PC1 > PC2 > ... > PCk (by variance)
```

**Visual intuition (2D → 1D):**
```
         x₂
          │      ●
          │  ● ●   ●           PC1 = direction of maximum variance
          │●   ●●    ●         PC2 = perpendicular to PC1
          │  ● ● ●●
          │●  ●  ●
          └──────────── x₁

          ●──────────────────── PC1 (captures most variance)
```

> 💡 "PCA asks: 'What are the directions in this data along which  
>      the data varies the most?' — and keeps only those directions."

---

## 🔍 When to Use PCA?

| Condition | Use PCA? |
|-----------|:--------:|
| High-dimensional data (many features) | ✅ Yes |
| Features are correlated | ✅ Yes — PCA decorrelates them |
| Need to visualize high-d data in 2D/3D | ✅ Yes |
| Preprocessing before clustering | ✅ Yes |
| Reduce noise from irrelevant dimensions | ✅ Yes |
| Features are independent (no correlation) | ❌ PCA won't help much |
| Need to preserve original feature names | ❌ PCA creates new components |
| Non-linear structure in data | ❌ → Use t-SNE, UMAP, Kernel PCA |

---

## 🧮 The Algorithm — Step by Step

```
Given: Data matrix X of shape (n, d)  — n samples, d features

Step 1: Center the data (subtract mean)
        X_centered = X − mean(X)
        (PCA requires zero-mean data)

Step 2: Compute Covariance Matrix
        C = (1/(n−1)) × X_centeredᵀ × X_centered   shape: (d, d)
        C[i,j] = covariance between feature i and feature j

Step 3: Eigendecomposition of C
        C × v = λ × v
        Eigenvalues  λ₁ ≥ λ₂ ≥ ... ≥ λd   (variance explained per PC)
        Eigenvectors v₁, v₂, ..., vd        (directions = principal components)

Step 4: Sort by eigenvalue (descending)
        PC1 = direction of maximum variance (largest λ₁)
        PC2 = direction of second-most variance (orthogonal to PC1)
        ...

Step 5: Project data onto top k components
        X_reduced = X_centered × [v₁ | v₂ | ... | vₖ]   shape: (n, k)

Output: k principal components capturing most variance
```

---

## 📐 Mathematical Foundations

### Covariance Matrix
```
C = (1/(n−1)) Xᵀ X

C[i,j] = cov(xᵢ, xⱼ)

Diagonal  = variance of each feature
Off-diagonal = covariance between features

Perfectly correlated features → large off-diagonals
Independent features         → diagonal matrix (no off-diagonals)
```

### Eigendecomposition
```
C = V Λ Vᵀ

Where:
  V = matrix of eigenvectors (columns = principal component directions)
  Λ = diagonal matrix of eigenvalues (λ₁ ≥ λ₂ ≥ ... ≥ λd)

Eigenvector vₖ = direction of k-th principal component
Eigenvalue  λₖ = variance explained along vₖ

Proportion of variance explained by PC k:
  PVE(k) = λₖ / Σⱼ λⱼ
```

### Singular Value Decomposition (SVD)
```
Equivalent to eigendecomposition but numerically more stable:

X = U Σ Vᵀ

Where:
  U = left singular vectors (n × n)
  Σ = singular values (diagonal, n × d)
  V = right singular vectors (d × d) = principal components

Eigenvalues: λₖ = σₖ² / (n−1)

sklearn uses SVD internally for numerical stability.
```

---

## 📊 Explained Variance

### Explained Variance Ratio

```
Each PC explains a fraction of total variance:

EVR(k) = λₖ / Σⱼ λⱼ

Example:
  PC1: λ₁ = 4.5 → EVR = 4.5/6.0 = 75%
  PC2: λ₂ = 1.2 → EVR = 1.2/6.0 = 20%
  PC3: λ₃ = 0.3 → EVR = 0.3/6.0 =  5%

→ PC1 + PC2 = 95% of total variance captured!

sklearn: pca.explained_variance_ratio_
```

### Cumulative Explained Variance

```
Cumulative EVR = Σₖ EVR(k)

Variance
  │████████████████████████████████████████████████████ 100%
  │████████████████████████████████████████████         95%
  │████████████████████████████████                     85%
  │████████████████████████                             75%
  └────────────────────────────────────── n_components
        1       2       3       4       5

Choose k where cumulative EVR ≥ 85–95%
```

---

## 🎯 Choosing Number of Components

### Method 1: Explained Variance Threshold
```
Keep k components that together explain ≥ 95% (or 85%, 90%) variance

pca = PCA(n_components=0.95)  # sklearn auto-selects k for 95%
```

### Method 2: Scree Plot (Elbow Method)
```
Plot eigenvalues vs component number:

Eigenvalue
│●
│ ●
│  ●●
│     ●●●●●●●●●●●●●  ← elbow here → keep components before this
└──────────────────── Component number

Choose k at the "elbow" — where eigenvalues drop sharply then flatten.
```

### Method 3: Kaiser's Rule
```
Keep all components with eigenvalue > 1.0
(Only components explaining more variance than a single original feature)

Applies when data is standardized (mean=0, std=1).
```

---

## 🔄 Reconstruction and Reconstruction Error

```
PCA allows approximate reconstruction of original data:

X_reconstructed = X_reduced × Vₖᵀ + mean(X)

Reconstruction Error = ||X − X_reconstructed||²

Lower error → better reconstruction → fewer dimensions lost
Higher k    → lower reconstruction error (trivially 0 when k=d)

Use case:
  Image compression: encode with k PCs, decode with reconstruction
  Anomaly detection: high reconstruction error → anomalous point
```

---

## 📌 PCA Loadings — Feature Contributions

```
Loadings = correlation between original features and principal components
         = eigenvectors scaled by √eigenvalue

Loading matrix V (d × k):
  V[i, j] = contribution of feature i to PC j

Interpretation:
  Large |loading| → feature i strongly influences PC j
  Sign of loading → direction of influence (positive/negative)

Biplot: plots both samples (scores) and features (loadings) simultaneously
```

---

## 🌀 Kernel PCA — Non-Linear Extension

```
Standard PCA: only finds linear structure

Kernel PCA:
  Apply kernel trick to map data to high-dimensional space
  Then apply PCA in that space
  → Captures non-linear structure

Kernels:
  'rbf'    → Gaussian kernel (most common)
  'poly'   → Polynomial kernel
  'cosine' → Cosine similarity

from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
```

---

## 🎛️ Key Hyperparameters

| Parameter | Effect | Typical Values |
|-----------|--------|----------------|
| `n_components` | Number of PCs to keep | Integer k, float (% variance), or 'mle' |
| `svd_solver` | SVD algorithm | 'auto', 'full', 'randomized' (fast) |
| `whiten` | Scale components to unit variance | False (default), True |
| `random_state` | Seed for randomized SVD | 42 |

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Not scaling before PCA | High-variance features dominate PCs | Always StandardScale first |
| Scaling after PCA | Data leakage if in pipeline | Scale → PCA, never PCA → Scale |
| Using PCA on target variable | Target is not a feature | Only apply to X, never y |
| Choosing k arbitrarily | Too many or too few components | Use explained variance threshold |
| PCA on non-linear data | Misses curved structure | Use Kernel PCA, t-SNE, or UMAP |
| Fitting PCA on test set | Data leakage | Fit on train, transform both |

---

## 🆚 PCA vs Other Reduction Methods

| Aspect | PCA | t-SNE | UMAP | Kernel PCA |
|--------|:---:|:-----:|:----:|:----------:|
| Linear | ✅ Yes | ❌ No | ❌ No | ❌ No |
| Preserves global structure | ✅ Yes | ❌ No | ⚠️ Partial | ✅ Yes |
| Preserves local structure | ⚠️ Partial | ✅ Yes | ✅ Yes | ✅ Yes |
| Reconstructable | ✅ Yes | ❌ No | ❌ No | ⚠️ Partial |
| Deterministic | ✅ Yes | ❌ No | ⚠️ Partial | ✅ Yes |
| Speed | ✅ Fast | ❌ Slow (O(n²)) | ✅ Fast | ⚠️ Medium |
| Best for | Preprocessing, compression | Visualization | Visualization | Non-linear |
| Works for prediction | ✅ Yes | ❌ No | ❌ No | ✅ Yes |

---

## 🔗 Related Topics

- `04_Unsupervised_Learning/KMeans` — Apply PCA before clustering
- `04_Unsupervised_Learning/DBSCAN` — PCA reduces curse of dimensionality
- `06_Feature_Selection` — PCA vs feature selection comparison
- `K_Nearest_Neighbors` — PCA before KNN for high-dimensional data
- `Support_Vector_Machine` — PCA for preprocessing high-d datasets

---

## 📚 References

- Scikit-learn `PCA`: [https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- Scikit-learn Decomposition Guide: [https://scikit-learn.org/stable/modules/decomposition.html](https://scikit-learn.org/stable/modules/decomposition.html)
- An Introduction to Statistical Learning — Chapter 12.2
- The Elements of Statistical Learning — Chapter 14.5
- Bishop, Pattern Recognition and Machine Learning — Chapter 12
