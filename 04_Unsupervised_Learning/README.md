# 🤖 Unsupervised Learning

## Overview

Unsupervised Learning is the branch of machine learning that works with **unlabeled data** —
the model explores the underlying structure, patterns, and relationships in the data **without**
any predefined target variable.

```
Raw Data (X)  →  Model Training  →  Discovered Structure  →  Clusters / Reduced Dimensions
```

This module covers the two fundamental unsupervised learning tasks — **Clustering** and
**Dimensionality Reduction** — with implementations for the most widely used algorithms in
industry and research.

---

## 📁 Folder Structure

```
04_Unsupervised_Learning/
│
├── Clustering/
│   ├── KMeans/
│   ├── Hierarchical_Clustering/
│   └── DBSCAN/
│
└── Dimensionality_Reduction/
    ├── PCA/
    ├── LDA/
    ├── tSNE/
    └── UMAP/
```

Each topic includes:

- 📘 `theory.md` → Concept explanation, formula, assumptions, pros & cons
- 📓 `.ipynb` → Interactive experiments, visualizations & evaluation
- 🐍 `.py` → Reusable, modular Python implementation *(where applicable)*

---

## 📌 Topics Covered

---

### 🧩 PART A — Clustering

Clustering algorithms group similar data points together based on their **inherent structure**,
with no prior knowledge of class labels.

---

#### 1. K-Means Clustering
A **centroid-based** algorithm that partitions data into K clusters by minimizing the
within-cluster variance.
- Random / K-Means++ centroid initialization
- Elbow method and Silhouette score for choosing K
- Assumes spherical, equally-sized clusters
- Sensitive to feature scaling and outliers

---

#### 2. Hierarchical Clustering
Builds a **tree of nested clusters** (dendrogram) using either an agglomerative (bottom-up) or
divisive (top-down) approach — no need to predefine the number of clusters.
- Linkage methods: Single, Complete, Average, Ward
- Distance metrics: Euclidean, Manhattan, Cosine
- Dendrogram cutting to determine cluster count
- Computationally expensive on large datasets

---

#### 3. DBSCAN (Density-Based Spatial Clustering)
A **density-based** algorithm that groups points that are closely packed together and marks
points in low-density regions as outliers.
- Core, border, and noise point classification
- Parameters: `eps` (neighborhood radius) and `min_samples`
- Discovers arbitrarily shaped clusters
- No need to specify number of clusters in advance
- Robust to outliers and noise

---

### 📉 PART B — Dimensionality Reduction

Dimensionality Reduction techniques reduce the number of features while preserving as much
meaningful information (structure, variance, or relationships) as possible.

---

#### 4. PCA (Principal Component Analysis)
A **linear** technique that projects data onto orthogonal axes (principal components) that
capture the maximum variance in the data.
- Eigenvalues and eigenvectors of the covariance matrix
- Explained variance ratio for choosing number of components
- Feature decorrelation and noise reduction
- Widely used for visualization and preprocessing

---

#### 5. LDA (Linear Discriminant Analysis)
A **supervised-flavored** linear technique that finds the axes that best separate known classes
— often used as a dimensionality reduction step before classification.
- Maximizes between-class variance, minimizes within-class variance
- Requires class labels (unlike PCA)
- Limited to (n_classes − 1) components
- Useful for class-separation-focused visualization

---

#### 6. t-SNE (t-Distributed Stochastic Neighbor Embedding)
A **non-linear** technique that preserves local neighborhood structure — excellent for
visualizing high-dimensional data in 2D/3D.
- Converts distances into probability distributions
- Perplexity parameter controls local vs global structure
- Computationally expensive, non-deterministic across runs
- Not suitable for reducing dimensions for downstream modeling

---

#### 7. UMAP (Uniform Manifold Approximation and Projection)
A **non-linear** manifold learning technique similar to t-SNE but faster and better at
preserving both local and global structure.
- Based on manifold learning and topological data analysis
- Parameters: `n_neighbors`, `min_dist`
- Faster and more scalable than t-SNE
- Can be used for both visualization and general-purpose dimensionality reduction

---

## 🎯 Learning Objectives

By completing this module, you will understand:

✔ The difference between **Clustering** and **Dimensionality Reduction** tasks  
✔ How each algorithm discovers structure in data — formula, objective, and optimization  
✔ When to use each algorithm based on data size, shape, and problem type  
✔ How to evaluate clustering results using appropriate metrics  
✔ How dimensionality reduction helps with **visualization**, **noise reduction**, and **preprocessing**  
✔ The difference between **linear** (PCA, LDA) and **non-linear** (t-SNE, UMAP) reduction techniques  
✔ How to implement and tune each algorithm using scikit-learn (and specialized libraries)  

---

## 📊 Algorithm Comparison at a Glance

### Clustering

| Algorithm | Cluster Shape | Needs K Upfront | Handles Outliers | Best For |
|-----------|:-------------:|:----------------:|:-----------------:|----------|
| K-Means | Spherical | ✅ | ❌ | Fast, general-purpose clustering |
| Hierarchical | Arbitrary | ❌ | ⚠️ | Small datasets, dendrogram insight |
| DBSCAN | Arbitrary | ❌ | ✅ | Noisy data, irregular cluster shapes |

### Dimensionality Reduction

| Algorithm | Linear/Non-Linear | Uses Labels | Preserves | Best For |
|-----------|:------------------:|:-----------:|:---------:|----------|
| PCA | Linear | ❌ | Global variance | Preprocessing, noise reduction |
| LDA | Linear | ✅ | Class separability | Supervised feature reduction |
| t-SNE | Non-linear | ❌ | Local structure | 2D/3D visualization |
| UMAP | Non-linear | ❌ | Local + global structure | Visualization & general reduction |

---

## 📏 Evaluation Metrics

### Clustering Metrics
| Metric | Notes |
|--------|-------|
| **Silhouette Score** | Measures cluster cohesion vs separation (-1 to 1) |
| **Davies-Bouldin Index** | Lower is better — ratio of within/between cluster distances |
| **Calinski-Harabasz Index** | Higher is better — variance ratio criterion |
| **Inertia (WCSS)** | Within-cluster sum of squares — used in K-Means elbow method |
| **Adjusted Rand Index (ARI)** | Compares clustering to ground truth labels (if available) |

### Dimensionality Reduction Metrics
| Metric | Notes |
|--------|-------|
| **Explained Variance Ratio** | Used in PCA — how much variance each component captures |
| **Reconstruction Error** | Difference between original and reconstructed data |
| **Trustworthiness / Continuity** | Measures how well local structure is preserved (t-SNE, UMAP) |
| **KL Divergence** | Used internally by t-SNE to measure embedding quality |

---

## 🛠 Tools & Libraries

- Python 3.x
- Pandas, NumPy
- Scikit-learn
- SciPy (hierarchical clustering, dendrograms)
- UMAP-learn (`umap-learn`)
- Matplotlib, Seaborn, Plotly (for visualization)

---

## 📌 Usage

Each folder contains:

- `theory.md` → Algorithm explanation, formula, and assumptions
- `.ipynb` → Full experiment with data, training, evaluation, and visualization
- `.py` → Clean, reusable implementation for production pipelines *(where applicable)*

### Recommended Workflow:
1. Read `theory.md` — understand the algorithm
2. Run `.ipynb` — experiment and visualize
3. Use `.py` — integrate into your ML pipeline *(where available)*

---

## 🚀 Importance in Machine Learning

Unsupervised Learning algorithms are crucial because:

- They uncover **hidden patterns and structure** in data without needing labels
- **Clustering** powers customer segmentation, anomaly detection, and market research
- **Dimensionality reduction** enables visualization of high-dimensional data and speeds up
  downstream modeling
- They're often used as a **preprocessing step** before supervised learning
- Understanding cluster shape and density assumptions drives better algorithm selection

---

## 📈 Recommended Learning Order

```
K-Means  →  Hierarchical Clustering  →  DBSCAN
        ↓
PCA  →  LDA  →  t-SNE  →  UMAP
```

---

## 📈 Next Steps

After completing this section, move to:

- `05_Model_Evaluation` → Cross-validation, ROC-AUC, confusion matrix
- `06_Feature_Selection` → Select features to improve model performance
- `07_Hyperparameter_Tuning` → GridSearchCV, RandomSearchCV, Bayesian Optimization
- `08_Ensemble_Learning` → Stacking, bagging, and advanced boosting

---

## 🤝 Contribution

This repository is part of a structured learning journey.  
Suggestions for improvements are always welcome.

---

## ⭐ Support

If you find this helpful, consider giving the repository a ⭐ on GitHub.
