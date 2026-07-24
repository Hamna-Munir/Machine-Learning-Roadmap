# 📘 Linear Discriminant Analysis (LDA) — Theory

---

## 📌 What is LDA?

Linear Discriminant Analysis (LDA) is a **supervised dimensionality reduction**  
and **classification technique** that finds the linear combinations of features  
that best **separate known classes** — maximizing between-class variance while  
minimizing within-class variance.

```
PCA (Unsupervised):   Finds directions of MAXIMUM TOTAL VARIANCE
                      → Ignores class labels

LDA (Supervised):     Finds directions of MAXIMUM CLASS SEPARATION
                      → Uses class labels to guide the projection
```

**Visual intuition:**
```
     x₂               x₂
      │   ●●            │   ●●
      │ ●●●●            │ ●●●●
      │  ●●●    ◆◆◆     │  ●●●◆◆◆
      │   ●   ◆◆◆◆◆     │   ●◆◆◆◆◆
      │       ◆◆◆       │      ◆◆◆
      └──────────── x₁   └──────────── x₁
                                 ↓
                         Project onto LDA axis
                ●●●●●●◆◆◆◆◆◆
                (well separated!)
```

> 💡 "LDA is a GPS for class separation — it tells you exactly  
>      which direction to look to best distinguish between groups."

---

## 🔍 When to Use LDA?

| Condition | Use LDA? |
|-----------|:--------:|
| Need supervised dimensionality reduction | ✅ Yes — primary strength |
| Classes are linearly separable | ✅ Yes |
| Want classification + reduction in one step | ✅ Yes |
| Features follow Gaussian distribution within classes | ✅ Yes |
| Multicollinear features | ✅ Yes — LDA handles via projection |
| Multiclass problem | ✅ Yes — up to K−1 discriminant components |
| Non-linear class boundaries | ❌ No → Kernel LDA or tree models |
| Very small dataset (< n features) | ⚠️ Caution — singular covariance matrix |
| Severely non-Gaussian features | ⚠️ Caution — assumption violated |

---

## 🧮 The Mathematics

### Objective Function

LDA maximizes the **Fisher criterion**:

```
J(w) = wᵀ Sᴮ w / wᵀ Sᵂ w

Where:
  w    = projection direction (discriminant axis)
  Sᴮ   = Between-class scatter matrix  (spread of class means)
  Sᵂ   = Within-class scatter matrix   (spread within each class)

Maximizing J(w):
  → Maximizes separation between class means (Sᴮ large)
  → Minimizes spread within each class (Sᵂ small)
  → Optimal direction: w* = Sᵂ⁻¹(μ₁ − μ₂)  for binary case
```

### Scatter Matrices

```
Within-class scatter (Sᵂ):
  Sᵂ = Σₖ Σᵢ∈class_k (xᵢ − μₖ)(xᵢ − μₖ)ᵀ

  → Sum of covariances within each class
  → Large Sᵂ → classes are spread out internally (hard to separate)

Between-class scatter (Sᴮ):
  Sᴮ = Σₖ nₖ (μₖ − μ)(μₖ − μ)ᵀ

  → Weighted sum of distances between class means and global mean
  → Large Sᴮ → class means are far apart (easy to separate)

Goal: maximize Sᴮ / Sᵂ
```

### Eigendecomposition Solution

```
The discriminant directions are eigenvectors of Sᵂ⁻¹Sᴮ:

  Sᵂ⁻¹Sᴮ w = λ w

Sort eigenvalues descending:
  λ₁ ≥ λ₂ ≥ ... ≥ λₘ

  → LD1 (first discriminant) = most class separation
  → LD2 (second discriminant) = second-most class separation (orthogonal to LD1)

Maximum number of discriminants:
  m = min(n_classes − 1, n_features)
  → Binary classification: only 1 discriminant axis (LD1)
  → 3 classes: up to 2 axes (LD1, LD2)
  → K classes: up to K−1 axes
```

---

## 📊 LDA vs PCA — Key Differences

```
                    PCA                         LDA
─────────────────────────────────────────────────────────────────
Type          Unsupervised                  Supervised
Objective     Max total variance            Max class separation
Uses labels?  ❌ No                         ✅ Yes
Max components min(n_features, n_samples−1) min(n_classes−1, n_features)
Best for      Compression, noise removal   Classification preprocessing
Assumption    None                         Gaussian, equal covariance
Output        Principal Components (PCs)   Linear Discriminants (LDs)
Projection    Directions of variance        Directions of discrimination
```

**When PCA beats LDA:**
```
Dataset with 3 classes:
  ●●●   ▲▲▲   ◆◆◆   (arranged in a line along the variance axis)

PCA: projects along variance → captures all three groups ✅
LDA: tries to maximize separation → may not align with variance axis
```

**When LDA beats PCA:**
```
Two overlapping classes:
  ●●●◆◆◆      → PCA keeps them overlapping (variance not aligned with separation)
  ↓ LDA
  ●●● | ◆◆◆   → LDA finds the separating axis ✅
```

---

## 🎯 LDA as a Classifier

LDA is also a direct **probabilistic classifier** using Bayes' theorem:

```
P(y=k | x) = P(x | y=k) × P(y=k) / P(x)

Assumptions:
  1. P(x | y=k) ~ Gaussian(μₖ, Σ)      (Gaussian within each class)
  2. All classes share the SAME covariance matrix Σ (homoscedastic)
     → Linear decision boundary (hence "Linear" DA)

Decision rule:
  Predict class k* = argmax_k [log P(y=k) − (1/2)(x−μₖ)ᵀ Σ⁻¹(x−μₖ)]

If classes have DIFFERENT covariances:
  → Quadratic Discriminant Analysis (QDA)
  → Quadratic decision boundaries
```

---

## 🔄 LDA vs QDA

```
LDA:
  Assumes equal covariance across all classes
  → Linear decision boundary
  → Fewer parameters → better with small datasets
  → from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

QDA:
  Estimates separate covariance per class
  → Quadratic decision boundary
  → More parameters → needs more data
  → from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

When to use QDA vs LDA:
  Small dataset      → LDA (fewer params, less overfitting)
  Large dataset      → QDA (can estimate per-class covariance)
  Equal covariances  → LDA
  Unequal covariances → QDA
```

---

## 📐 Explained Variance in LDA

```
Each discriminant direction (LD) explains a fraction of class separation:

EVR(k) = λₖ / Σⱼ λⱼ

lda.explained_variance_ratio_

Unlike PCA (where cumulative EVR reaches 100%), LDA's EVR is bounded
by the number of discriminant directions (K−1 max).
```

---

## 🎛️ Key Hyperparameters

| Parameter | Effect | Values |
|-----------|--------|--------|
| `n_components` | Number of discriminant directions to keep | 1 to min(K−1, n_features) |
| `solver` | Algorithm to solve the eigenvalue problem | 'svd' (default), 'lsqr', 'eigen' |
| `shrinkage` | Regularization for covariance estimation | None, 'auto' (Ledoit-Wolf), float [0,1] |
| `priors` | Class prior probabilities | None (estimated from data) or array |
| `tol` | Threshold for rank estimation in SVD solver | 1e-4 (default) |

### Shrinkage — Regularized LDA

```
Problem: Sᵂ can be singular (non-invertible) when:
  - n_samples < n_features (more features than samples)
  - Features are perfectly correlated

Solution: Shrinkage regularization
  Σ_shrunk = (1 − α) × Σ_sample + α × trace(Σ_sample)/d × I

  α = 0: standard sample covariance (no shrinkage)
  α = 1: fully diagonal (spherical) covariance
  α = 'auto': Ledoit-Wolf optimal shrinkage

sklearn: LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr')
```

---

## ⚠️ Assumptions and Violations

| Assumption | Description | Effect if violated |
|------------|-------------|-------------------|
| Gaussian features | Each class's features are normally distributed | Poor probability estimates |
| Equal covariance | All classes have same covariance matrix Σ | Suboptimal boundary; use QDA |
| Independence | Observations are independent | Inflated confidence |
| No perfect multicollinearity | Features not perfectly linearly dependent | Singular Sᵂ → use shrinkage |

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Not scaling before LDA | Dominated by large-range features | Always StandardScale first |
| Using LDA for unsupervised tasks | Requires class labels | Use PCA instead |
| Expecting > K−1 components | LDA max = K−1 | Understand the theoretical limit |
| Singular covariance matrix | Small n vs large d | Use shrinkage='auto' |
| Non-Gaussian features | Assumption violated | Check distributions; may still work |
| QDA on small dataset | Too many parameters → overfitting | Use LDA or shrinkage |

---

## 🆚 LDA vs PCA vs t-SNE

| Aspect | LDA | PCA | t-SNE |
|--------|:---:|:---:|:-----:|
| Supervised | ✅ Yes | ❌ No | ❌ No |
| Linear | ✅ Yes | ✅ Yes | ❌ No |
| Classification | ✅ Yes | ❌ No | ❌ No |
| Visualization | ✅ (K−1 axes) | ✅ Any k | ✅ Best |
| Max components | K−1 | min(n,d) | 2–3 |
| Speed | ✅ Fast | ✅ Fast | ❌ Slow |
| New data | ✅ Yes | ✅ Yes | ❌ No |

---

## 🔗 Related Topics

- `PCA` — Unsupervised counterpart; compare with LDA
- `Logistic_Regression` — Also a linear classifier; different objective
- `Support_Vector_Machine` — Finds max-margin boundary (not max class sep)
- `04_Unsupervised_Learning/PCA` — Compare supervised vs unsupervised reduction
- `06_Feature_Selection` — LDA as embedded feature extractor

---

## 📚 References

- Scikit-learn `LinearDiscriminantAnalysis`: [https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
- Scikit-learn `QuadraticDiscriminantAnalysis`: [https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html)
- Original Fisher LDA Paper (Fisher, 1936)
- An Introduction to Statistical Learning — Chapter 4.4
- The Elements of Statistical Learning — Chapter 4.3
