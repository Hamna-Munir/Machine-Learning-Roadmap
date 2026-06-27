# 📘 Random Forest — Theory

---

## 📌 What is Random Forest?

Random Forest is an **ensemble learning method** that builds **many Decision Trees**  
during training and combines their predictions — using **majority vote** for classification  
or **averaging** for regression. It is one of the most powerful and widely used  
general-purpose machine learning algorithms.

```
Training Data
     │
     ├──► Bootstrap Sample 1 → Tree 1  ──┐
     ├──► Bootstrap Sample 2 → Tree 2  ──┤
     ├──► Bootstrap Sample 3 → Tree 3  ──┼──► Majority Vote / Average → Final Prediction
     ├──► ...                            │
     └──► Bootstrap Sample N → Tree N  ──┘
```

> 💡 "A single Decision Tree is a brilliant but unstable expert.  
>      A Random Forest is a committee of hundreds of slightly-different experts —  
>      individually imperfect, but collectively far more reliable."

---

## 🔍 When to Use Random Forest?

| Condition | Use Random Forest? |
|-----------|:------------------:|
| Tabular data, general-purpose classification/regression | ✅ Yes — excellent default choice |
| Need robustness to overfitting | ✅ Yes |
| Need feature importance ranking | ✅ Yes |
| Mixed categorical and numerical features | ✅ Yes |
| Need extreme interpretability | ❌ No → Use single Decision Tree |
| Need fastest possible predictions | ⚠️ Caution → slower than single tree |
| Very high-dimensional sparse data (e.g., text) | ❌ No → Use Naive Bayes / Linear models |

---

## 🧮 Two Core Techniques

Random Forest combines **two sources of randomness** to build diverse trees:

### 1️⃣ Bootstrap Aggregating (Bagging)

```
For each of N trees:
  1. Draw a random sample of size n WITH REPLACEMENT from training data
     (this is called a "bootstrap sample")
  2. Train a Decision Tree on this bootstrap sample
  3. Some original samples appear multiple times; ~37% are never selected
     (these unselected samples are called "Out-of-Bag" or OOB samples)

Why it works:
  Each tree sees a slightly different version of the data
  → Trees make different errors
  → Averaging/voting cancels out individual errors
  → Reduces VARIANCE without increasing bias
```

### 2️⃣ Random Feature Selection (Feature Bagging)

```
At each split in each tree:
  Instead of considering ALL features, randomly select a SUBSET of features
  Choose the best split only from this subset

Default subset size:
  Classification: max_features = √(total_features)
  Regression:     max_features = total_features / 3

Why it works:
  Prevents all trees from always splitting on the same dominant feature
  → Forces trees to be more DIVERSE
  → Decorrelates trees → further reduces variance
```

---

## 🎯 Why Random Forest Reduces Variance

```
Single Decision Tree:
  High variance — different training samples produce very different trees

Random Forest with N trees:
  Variance of average ≈ Variance(single tree) / N   (if trees were independent)

  But trees are NOT fully independent (same underlying data)
  → Feature randomness REDUCES correlation between trees
  → Lower correlation → variance reduction is closer to the ideal 1/N

Mathematically:
  Var(average) = ρσ² + (1-ρ)σ²/N

  Where:
    ρ = correlation between trees
    σ² = variance of a single tree

  Lower ρ (more diverse trees) → lower overall variance ✅
```

---

## 🗳️ Aggregation — How Predictions Combine

### Classification
```
Each tree votes for a class
Final prediction = majority vote (mode)

Or, using probabilities:
  P(class=k) = (1/N) × Σ P_tree_i(class=k)
  Final prediction = argmax of averaged probabilities
```

### Regression
```
Final prediction = average of all tree predictions

ŷ = (1/N) × Σ ŷ_tree_i
```

---

## 📊 Out-of-Bag (OOB) Error Estimation

A unique advantage of Random Forest — **free validation without a separate test set**:

```
For each tree, ~37% of training samples were NOT used (OOB samples)
  → Each sample is OOB for roughly 1/3 of all trees

OOB prediction for sample i:
  Use only the trees where sample i was OOB
  → Aggregate their predictions
  → Compare to true label

OOB Error ≈ Test Error (without needing a separate holdout set!)

sklearn: RandomForestClassifier(oob_score=True)
         model.oob_score_  →  OOB accuracy
```

---

## 🎛️ Key Hyperparameters

| Parameter | Effect | Typical Values |
|-----------|--------|-----------------|
| `n_estimators` | Number of trees | 100–1000 (more = better, but slower) |
| `max_depth` | Max depth per tree | None, 5–30 |
| `max_features` | Features considered per split | 'sqrt', 'log2', None |
| `min_samples_split` | Min samples to split a node | 2–20 |
| `min_samples_leaf` | Min samples per leaf | 1–10 |
| `bootstrap` | Whether to use bagging | True (default) |
| `oob_score` | Compute out-of-bag score | True/False |
| `class_weight` | Handle imbalanced classes | 'balanced', None |
| `n_jobs` | Parallel processing | -1 (use all cores) |

---

## 🎯 Feature Importance — Two Methods

### 1. Mean Decrease in Impurity (MDI) — Default

```
For each feature:
  Sum the (weighted) impurity decrease at every node where it was used,
  averaged across all trees in the forest

⚠️ Biased toward high-cardinality features (many unique values)
⚠️ Computed on training data — can be misleading

sklearn: model.feature_importances_
```

### 2. Permutation Importance — More Reliable

```
For each feature:
  1. Measure baseline model performance
  2. Randomly shuffle that feature's values (breaking its relationship with y)
  3. Measure performance drop
  4. Larger drop = more important feature

✅ More reliable than MDI — measured on test/validation data
✅ Not biased toward high-cardinality features

sklearn: from sklearn.inspection import permutation_importance
```

---

## 📈 N_Estimators — How Many Trees?

```
Error
│●
│ ●
│  ●●
│    ●●●
│        ●●●●●●●●●●●●●●●●●●●●●●●  ← plateaus, diminishing returns
└─────────────────────────────────── n_estimators
  10  50  100  200  500  1000

More trees:
  ✅ Always reduces variance (never increases overfitting from MORE trees)
  ✅ Error plateaus — more trees beyond the plateau just costs compute time
  ❌ Diminishing returns after a certain point (typically 100-500)

Rule of thumb: Start with 100-200 trees, increase if compute budget allows
```

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Using MDI importance for high-cardinality features | Biased ranking | Use permutation importance instead |
| Too few trees | High variance remains | Use at least 100 trees |
| max_depth=None on noisy data | Individual trees overfit | Still works due to averaging, but tune for speed |
| Ignoring OOB score | Wasting free validation signal | Set oob_score=True |
| Treating RF as fully interpretable | Hundreds of trees, no single rule | Use feature importance, not individual tree rules |
| Class imbalance ignored | Biased toward majority class | Use class_weight='balanced' |

---

## 🆚 Random Forest vs Other Models

| Aspect | Random Forest | Decision Tree | Gradient Boosting | XGBoost |
|--------|:--------------:|:-------------:|:------------------:|:-------:|
| Variance | ✅ Low | ❌ High | ✅ Low | ✅ Low |
| Bias | ⚠️ Medium | ⚠️ Medium | ✅ Low | ✅ Low |
| Training | Parallel (fast) | Fast | Sequential (slower) | Optimized parallel |
| Overfitting Risk | ✅ Low | ❌ High | ⚠️ Medium (needs tuning) | ⚠️ Medium |
| Interpretability | ⚠️ Medium | ✅ High | ❌ Low | ❌ Low |
| Typical Accuracy | ✅ Good | ⚠️ OK | ✅ Very Good | ✅ Excellent |

---

## 🔗 Related Topics

- `Decision_Trees` — The base learner used in Random Forest
- `Gradient_Boosting` — Sequential ensemble alternative
- `XGBoost` / `LightGBM` / `CatBoost` — Optimized boosting frameworks
- `08_Ensemble_Learning` — Bagging, boosting, and stacking theory
- `06_Feature_Selection` — Use RF importance for feature ranking

---

## 📚 References

- Scikit-learn `RandomForestClassifier`: [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- Original Random Forest Paper (Breiman, 2001): [https://link.springer.com/article/10.1023/A:1010933404324](https://link.springer.com/article/10.1023/A:1010933404324)
- Permutation Importance: [https://scikit-learn.org/stable/modules/permutation_importance.html](https://scikit-learn.org/stable/modules/permutation_importance.html)
- An Introduction to Statistical Learning — Chapter 8.2
- The Elements of Statistical Learning — Chapter 15
