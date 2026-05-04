# 📘 Multivariate Analysis — Theory

---

## 📌 What is Multivariate Analysis?

Multivariate Analysis is the **simultaneous examination of three or more variables**  
to understand complex interactions, patterns, and structures within data.

```
Univariate   →  One variable         (What does X look like?)
Bivariate    →  Two variables        (How does X relate to Y?)
Multivariate →  Three or more        (How do X, Y, Z interact together?)
```

> 💡 "Real-world phenomena are rarely explained by a single variable.  
>      Multivariate analysis reveals the hidden structure behind complex data."

---

## 🔍 Why Multivariate Analysis?

| Goal | What It Reveals |
|------|----------------|
| **Interaction effects** | Does the effect of X on Y change depending on Z? |
| **Confounding variables** | Is a bivariate relationship explained by a third variable? |
| **Cluster structure** | Do natural groupings exist in high-dimensional space? |
| **Dimensionality** | Which combinations of features explain the most variance? |
| **Feature redundancy** | Are multiple features capturing the same information? |
| **Model readiness** | How separable are the classes in feature space? |

---

## 🏷️ Types of Multivariate Techniques

```
Multivariate Analysis
    │
    ├── 📊 Visualization Techniques
    │       ├── Pair Plot (Scatter Matrix)
    │       ├── Hue-Colored Scatter Plot
    │       ├── 3D Scatter Plot
    │       ├── Parallel Coordinates Plot
    │       ├── Radar / Spider Chart
    │       └── Bubble Chart
    │
    ├── 📐 Statistical Techniques
    │       ├── Correlation Matrix (all pairs)
    │       ├── Variance Inflation Factor (VIF)
    │       └── Multivariate ANOVA (MANOVA)
    │
    └── 🤖 Dimensionality Reduction (for visualization)
            ├── PCA  (Principal Component Analysis)
            ├── t-SNE (t-Distributed Stochastic Neighbor Embedding)
            └── UMAP (Uniform Manifold Approximation & Projection)
```

---

## 🛠️ Visualization Techniques

---

### 1️⃣ Pair Plot (Scatter Matrix)

Plots **every numerical variable against every other** in a grid — diagonal shows the  
univariate distribution of each variable.

```
         Age      Salary    Score
Age    [hist]  [scatter] [scatter]
Salary [scat]   [hist]   [scatter]
Score  [scat]  [scatter]   [hist]

Diagonal   → histogram or KDE of each variable
Off-diagonal → scatter plot of every pair
```

**Enhancements:**
- `hue` parameter → color points by a categorical variable to reveal class separation
- `kind='reg'` → adds regression lines to each scatter
- `diag_kind='kde'` → smooth KDE on diagonal instead of histogram

**What to look for:**
- Linear trends between pairs (positive/negative correlation)
- Clusters of points that separate by class
- Outliers isolated from the main cloud
- Non-linear relationships (curved patterns)

---

### 2️⃣ Hue-Colored Scatter Plot

A **bivariate scatter plot with a third variable** encoded as color (and optionally size or shape).

```
Y
│   ● ●  ○         ● = Class A (blue)
│  ● ●   ○ ○       ○ = Class B (red)
│ ●    ○ ○
│      ○ ○ ○
└──────────────── X

Color encodes the 3rd variable (categorical or numerical)
```

**Extended encoding options:**
- `hue` → 3rd variable as color
- `size` → 4th variable as point size (bubble chart)
- `style` → 5th variable as marker shape

**When to use:**
- Checking if class separation exists in 2D feature space
- Visualizing interaction between two numerical and one categorical variable

---

### 3️⃣ 3D Scatter Plot

Extends scatter to **three numerical dimensions** simultaneously.

```
        Z
        │     ●
        │  ●     ●
        │     ●
        └────────── Y
       /
      X
```

**Advantages:**
- Shows true 3D structure in one view
- Useful when two 2D projections each miss part of the pattern

**Limitations:**
- Hard to read when projected onto a 2D screen
- Overplotting is worse than 2D
- Use rotation (interactive plots with Plotly) for best results

---

### 4️⃣ Parallel Coordinates Plot

Each **vertical axis represents one variable** — each line represents one observation,  
connecting its values across all dimensions.

```
   Age    Salary   Score   Exp    Churn
    │       │        │       │      │
70  ─       ─        ─       ─      ─  1
    │   ╱   │    ╲   │   ╱   │      │
    │  ╱    │     ╲  │  ╱    │      │
35  ─       ─        ─       ─      ─  0
    │       │        │       │      │
18  ─       ─        ─       ─      ─
   Age    Salary   Score   Exp    Churn
```

**What to look for:**
- Lines of the same color that **follow a similar path** → shared pattern
- **Crossings** between axes → negative correlation
- **No crossings** between axes → positive correlation
- Clusters of parallel lines → a subgroup in the data

**Best for:**
- High-dimensional data overview (many features at once)
- Identifying feature profiles for each class

---

### 5️⃣ Radar / Spider Chart

Displays **multiple variables on radial axes** from a central point — each observation  
or group becomes a polygon.

```
          Score
           │
    ────── │ ──────
   /       │       \
  /   ████ │ ████   \
Age ██████ ● ██████ Salary
  \   ████ │ ████   /
   \       │       /
    ────── │ ──────
           │
        Experience
```

**When to use:**
- Comparing **profiles across groups** (e.g., average feature values for Churn=0 vs Churn=1)
- Communicating multi-dimensional group characteristics to stakeholders
- Dashboards and reports

**Limitations:**
- Hard to read with many variables (> 8 axes)
- Area can be misleading due to axis ordering

---

### 6️⃣ Bubble Chart

A scatter plot where a **third numerical variable** is encoded as the **size** of each point.

```
Y
│      ◉              ◉ = large value of Z
│  ◉       ●
│       ◉       ●     ● = small value of Z
│   ●
└──────────────── X

X = Feature 1
Y = Feature 2
Size = Feature 3 (e.g., Income)
Color = Feature 4 (e.g., Education level)
```

---

## 🛠️ Statistical Techniques

---

### 7️⃣ Variance Inflation Factor (VIF)

Measures how much the variance of a regression coefficient is **inflated due to  
multicollinearity** with other features.

**Formula:**
```
VIF(Xⱼ) = 1 / (1 − R²ⱼ)

Where R²ⱼ = R-squared of regressing Xⱼ on all other features
```

**Interpretation:**
```
VIF = 1        → No multicollinearity
VIF = 1–5      → Low multicollinearity  (acceptable)
VIF = 5–10     → Moderate               (investigate)
VIF > 10       → High multicollinearity  (problematic — consider dropping)
```

**When to use:**
- Before building **linear regression** models
- After seeing high pairwise correlations in the correlation heatmap

---

### 8️⃣ Covariance Matrix

Measures how two variables **vary together** — the multivariate generalization of variance.

**Formula:**
```
Cov(X, Y) = Σ(xᵢ − x̄)(yᵢ − ȳ) / (n − 1)

Positive Cov(X,Y) → X and Y increase together
Negative Cov(X,Y) → X increases as Y decreases
Zero Cov(X,Y)     → No linear relationship
```

**Note:** Covariance is scale-dependent — normalize it to get the **correlation coefficient**.

---

## 🤖 Dimensionality Reduction Techniques

---

### 9️⃣ PCA (Principal Component Analysis)

Finds the **orthogonal directions of maximum variance** in the data and projects  
the data onto a lower-dimensional space.

```
Original Space (3D)         PCA Space (2D)
     Z                        PC2
     │  ●●                     │   ●●
     │●   ●                   ●│     ●
     │  ●   ─── PCA ──►       ─┤──────── PC1
     └──────── Y              ●│  ●●
    /                          │
   X
```

**Key concepts:**
- **Principal Components (PCs):** New axes that maximize variance
- **Explained Variance Ratio:** How much of total variance each PC captures
- **Scree Plot:** Shows variance explained per component → helps choose how many PCs to keep

```
Scree Plot:
Variance %
    │  ●
40% ─     ●
    │        ●
10% ─           ● ● ● ●
    └──────────────────── PC1  PC2  PC3  PC4 ...

Elbow point → number of components to keep
```

**When to use:**
- High-dimensional data (many correlated features)
- Visualization of high-dimensional data in 2D/3D
- Removing multicollinearity before linear models

**⚠️ Limitations:**
- Components are linear combinations — not individually interpretable
- Sensitive to feature scale → **always standardize before PCA**

---

### 🔟 t-SNE (t-Distributed Stochastic Neighbor Embedding)

A **non-linear dimensionality reduction** technique that preserves **local structure** —  
nearby points in high-dimensional space remain nearby in 2D.

**Core Idea:**
```
High-dimensional space:   ●●●        ○○○
                          ●●●        ○○○
                               ■■■

t-SNE 2D projection:    [●●●]   [○○○]
                              [■■■]

Clusters in high-D → Clusters in 2D ✅
```

**Key parameters:**
- `perplexity` → balance between local and global structure (typical: 5–50)
- `n_iter` → number of optimization iterations (minimum: 250, recommended: 1000)

**⚠️ Important caveats:**
- t-SNE is **stochastic** — different runs give different layouts
- **Distances between clusters are NOT meaningful** — only cluster membership matters
- **Cluster sizes are NOT meaningful** — t-SNE inflates small clusters
- Only use for **visualization**, never for feature engineering

---

### 1️⃣1️⃣ UMAP (Uniform Manifold Approximation & Projection)

A **faster, more scalable** alternative to t-SNE that better preserves **global structure**.

**Comparison:**
```
Feature          t-SNE               UMAP
─────────────────────────────────────────────
Speed            Slow                Fast ✅
Global structure Poor                Better ✅
Local structure  Excellent ✅        Good
Deterministic    No                  Yes (with seed) ✅
Scalability      Limited             Large datasets ✅
```

**When to use over t-SNE:**
- Large datasets (> 10,000 rows)
- When global cluster relationships matter
- When reproducibility is required

---

## 📊 Technique Selection Guide

```
What is your goal?
    │
    ├── Visualize all feature pairs at once
    │       └── Pair Plot (hue by target)
    │
    ├── Show 3 variables simultaneously
    │       ├── Hue-colored scatter (2 num + 1 cat)
    │       └── Bubble chart (3 numerical)
    │
    ├── Show many variables in one view
    │       ├── Parallel Coordinates Plot
    │       └── Radar / Spider Chart
    │
    ├── Check for multicollinearity
    │       ├── Correlation Matrix Heatmap
    │       └── VIF Analysis
    │
    └── Visualize high-dimensional structure
            ├── Balanced speed + quality   → PCA (2D projection)
            ├── Best cluster separation    → t-SNE
            └── Large dataset / global    → UMAP
```

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| PCA without scaling | Features with large range dominate | Always StandardScale before PCA |
| Interpreting t-SNE distances | Distances between clusters not meaningful | Only trust cluster membership |
| Too many pair plot features | Visual clutter, unreadable | Limit to 6–8 most important features |
| Ignoring VIF | Multicollinearity hidden from view | Always compute VIF before linear models |
| Radar charts with > 8 axes | Becomes unreadable | Limit axes or use parallel coordinates instead |
| t-SNE with low n_iter | Poor convergence | Use n_iter ≥ 1000 |
| Dropping PCA components blindly | May lose signal | Check cumulative variance (aim for 90–95%) |

---

## 🔗 Related Topics

- `Bivariate_Analysis` — Foundation before extending to 3+ variables
- `Correlation_Analysis` — Pairwise correlation matrix across all features
- `04_Unsupervised_Learning / PCA` — Deep-dive mathematical treatment of PCA
- `04_Unsupervised_Learning / tSNE` — Full t-SNE implementation and tuning
- `04_Unsupervised_Learning / UMAP` — Full UMAP implementation
- `06_Feature_Selection` — Use VIF and correlation to select features

---

## 📚 References

- Seaborn Pair Plot: [https://seaborn.pydata.org/generated/seaborn.pairplot.html](https://seaborn.pydata.org/generated/seaborn.pairplot.html)
- Sklearn PCA: [https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- Sklearn t-SNE: [https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
- Statsmodels VIF: [https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html](https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html)
- UMAP Documentation: [https://umap-learn.readthedocs.io/](https://umap-learn.readthedocs.io/)
