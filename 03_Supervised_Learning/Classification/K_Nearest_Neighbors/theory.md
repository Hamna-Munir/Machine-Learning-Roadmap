# 📘 K-Nearest Neighbors (KNN) — Theory

---

## 📌 What is KNN?

K-Nearest Neighbors is a **non-parametric, instance-based (lazy) learning** algorithm  
that classifies a new data point based on the **majority class** among its K closest  
neighbors in the training set — or averages their values for regression.

```
Training phase:  Store all training data  (no learning happens!)
Prediction phase: For each test point:
  1. Compute distance to all training points
  2. Select K nearest neighbors
  3. Return majority class (classification) or mean (regression)
```

> 💡 "KNN is the simplest ML algorithm — it literally just says:  
>      'Show me your neighbors and I'll tell you who you are.'"

---

## 🔍 When to Use KNN?

| Condition | Use KNN? |
|-----------|:--------:|
| Small to medium dataset (< 50K rows) | ✅ Yes |
| Non-linear, complex decision boundary | ✅ Yes |
| Low dimensionality (< 20 features) | ✅ Yes |
| Need simple baseline classifier | ✅ Yes |
| Large dataset (> 100K rows) | ❌ No — too slow at prediction |
| High-dimensional data (curse of dimensionality) | ❌ No |
| Need interpretable model | ❌ No — black box |
| Need fast predictions | ❌ No — O(n) per prediction |

---

## 🧮 The Algorithm

### Step-by-Step Prediction

```
Given: Training set {(x₁,y₁), ..., (xₙ,yₙ)}, new point xₜ, hyperparameter K

Step 1: Compute distance d(xₜ, xᵢ) for all i = 1..n
Step 2: Sort distances in ascending order
Step 3: Select the K points with smallest distances → K nearest neighbors
Step 4a (Classification): ŷ = mode(y₁, y₂, ..., yₖ)  ← majority vote
Step 4b (Regression):     ŷ = mean(y₁, y₂, ..., yₖ)  ← average
```

**Visual Example (K=3):**
```
       ■ ■                  ■ = Class A
   ● ■     ■                ● = Class B
       ★                    ★ = New point (which class?)
   ●     ●
     ●

K=3: 3 nearest neighbors of ★ are: ■, ■, ●
Majority vote: 2 × Class A, 1 × Class B → Predict Class A ✅
```

---

## 📏 Distance Metrics

The choice of distance metric fundamentally shapes which neighbors are "near":

### 1. Euclidean Distance (L2) — Default

```
d(x, y) = √(Σᵢ (xᵢ − yᵢ)²)

Best for: Continuous features, isotropic data
Sensitive to scale → MUST standardize before KNN
```

### 2. Manhattan Distance (L1)

```
d(x, y) = Σᵢ |xᵢ − yᵢ|

Best for: High-dimensional data, grid-like movement
More robust to outliers than Euclidean
```

### 3. Minkowski Distance (General)

```
d(x, y) = (Σᵢ |xᵢ − yᵢ|ᵖ)^(1/p)

p = 1 → Manhattan distance
p = 2 → Euclidean distance
p → ∞ → Chebyshev distance (max difference)
```

### 4. Hamming Distance

```
d(x, y) = (1/n) × Σᵢ 𝟙[xᵢ ≠ yᵢ]

Best for: Categorical features (counts mismatches)
```

---

## 🎛️ The K Hyperparameter

Choosing K is the most critical decision in KNN:

```
Small K (e.g., K=1):
  + Very flexible — captures local patterns
  + Low bias
  − High variance — sensitive to noise and outliers
  − Overfits to training data

Large K (e.g., K=n):
  + Very smooth boundary
  + Low variance
  − High bias — may miss important local patterns
  − Underfits — predicts majority class for all points

Decision boundary:
K=1     K=5       K=20      K=100
[jagged] [smooth]  [smoother] [nearly flat]
```

**Rule of thumb:** Start with `K = √n` where n = number of training samples.  
Always use **odd K** for binary classification to avoid ties.

---

## 🔄 Weighted KNN

Assign **weights to neighbors** based on their distance — closer neighbors vote more:

```
Standard KNN:  All K neighbors vote equally (weight = 1)
Weighted KNN:  Weight = 1 / d(xₜ, xᵢ)²  (closer = higher weight)

sklearn: KNeighborsClassifier(weights='uniform')  ← default
         KNeighborsClassifier(weights='distance') ← weighted
```

**When to use weighted KNN:**
- When the K-th neighbor is much farther than the 1st neighbor
- When boundary points matter more than interior points

---

## 📐 Curse of Dimensionality

KNN suffers severely in high dimensions:

```
In high dimensions:
  - All points become approximately equidistant from each other
  - The concept of "nearest" loses meaning
  - Exponentially more data needed to maintain the same density

Example:
  In 1D: Need n points to cover [0,1] with resolution ε
  In d dimensions: Need n^d points for the same resolution

Rule: KNN works best when dimensionality d ≤ 20
Solution for high-d: Apply PCA/t-SNE first, then KNN
```

---

## ⚡ Computational Complexity

| Phase | Brute Force | KD-Tree | Ball Tree |
|-------|:-----------:|:-------:|:---------:|
| Training | O(1) | O(n log n) | O(n log n) |
| Prediction (per point) | O(n×d) | O(log n × d) | O(log n × d) |
| Best for | Small n | Low-d | High-d/non-Euclidean |

```python
# sklearn algorithms:
KNeighborsClassifier(algorithm='brute')    # exact, slow for large n
KNeighborsClassifier(algorithm='kd_tree')  # fast for low-d (d < 20)
KNeighborsClassifier(algorithm='ball_tree') # better for high-d
KNeighborsClassifier(algorithm='auto')     # sklearn chooses automatically
```

---

## 📊 KNN for Regression

The same algorithm — instead of majority vote, KNN returns the **mean of K neighbors**:

```
ŷ = (1/K) × Σₖ yₖ   (uniform weights)
ŷ = Σₖ (wₖ × yₖ) / Σₖ wₖ   (distance weights)

from sklearn.neighbors import KNeighborsRegressor
```

---

## 🔧 Feature Scaling — Critical for KNN

Distance-based algorithms are extremely sensitive to feature scale:

```
Without scaling:
  Feature A: [0, 1000]  (e.g., salary)
  Feature B: [0, 1]     (e.g., score)

  Euclidean distance is dominated almost entirely by Feature A
  → Feature B has near-zero influence on which neighbors are selected ❌

With StandardScaler:
  Both features in range [-3, 3]
  → Equal contribution to distance ✅
```

**Always apply StandardScaler (or MinMaxScaler) before KNN.**

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Not scaling features | Distance dominated by large-range features | Always StandardScale first |
| Even K for binary classification | Ties in voting | Use odd K |
| K=1 in production | Sensitive to noise and outliers | Use K≥5 with CV |
| Too large K | Oversmoothing, underfitting | Use validation curve to find optimal K |
| High-dimensional data | Curse of dimensionality | Apply PCA first or use different algorithm |
| Large dataset | O(n) prediction time is too slow | Use approximate neighbors or switch algorithm |
| Using KNN with categorical features | Manhattan/Euclidean not meaningful | Encode + scale, or use Hamming distance |

---

## 🆚 KNN vs Other Classifiers

| Aspect | KNN | Logistic Reg. | Decision Tree | SVM |
|--------|:---:|:-------------:|:-------------:|:---:|
| Training Time | ✅ Instant | ✅ Fast | ✅ Fast | ❌ Slow |
| Prediction Time | ❌ Slow (O(n)) | ✅ Fast | ✅ Fast | ✅ Fast |
| Interpretability | ❌ Black box | ✅ High | ✅ High | ❌ Low |
| Non-linear boundary | ✅ Yes | ❌ No | ✅ Yes | ✅ (kernel) |
| Feature Scaling | ✅ Required | ✅ Required | ❌ Not needed | ✅ Required |
| Works on small data | ✅ Yes | ⚠️ OK | ✅ Yes | ✅ Yes |
| Noise sensitivity | ❌ High | ✅ Low | ⚠️ Medium | ✅ Low |

---

## 🔗 Related Topics

- `Logistic_Regression` — Linear alternative for classification
- `Support_Vector_Machine` — Also distance-based, but finds optimal margin
- `04_Unsupervised_Learning/KMeans` — Also distance-based, but for clustering
- `06_Feature_Selection` — Reduce dimensions before KNN
- `07_Hyperparameter_Tuning` — GridSearchCV for optimal K

---

## 📚 References

- Scikit-learn `KNeighborsClassifier`: [https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- Scikit-learn Nearest Neighbors Guide: [https://scikit-learn.org/stable/modules/neighbors.html](https://scikit-learn.org/stable/modules/neighbors.html)
- An Introduction to Statistical Learning — Chapter 2.2
- The Elements of Statistical Learning — Chapter 13
