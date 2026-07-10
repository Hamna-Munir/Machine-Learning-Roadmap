# 📘 LightGBM — Theory

---

## 📌 What is LightGBM?

LightGBM (**Light Gradient Boosting Machine**) is a **histogram-based gradient  
boosting framework** developed by Microsoft Research. It achieves significantly  
faster training speeds and lower memory usage than XGBoost while maintaining  
comparable or better accuracy — making it the preferred choice for large datasets.

```
Standard GBM / XGBoost:
  For each split: scan ALL data points for ALL features
  Complexity: O(#data × #features) per tree level
  → Slow for large n and high-dimensional data

LightGBM:
  Step 1: Bin continuous features into discrete histograms (~256 bins)
  Step 2: For each split: scan histograms, NOT raw data
  Complexity: O(#bins × #features) per split — much faster!
  → GOSS + EFB reduce data and features further
```

> 💡 "LightGBM is XGBoost's faster sibling — it trains in minutes  
>      what XGBoost takes hours to process on large datasets."

---

## 🔍 When to Use LightGBM?

| Condition | Use LightGBM? |
|-----------|:------------:|
| Large datasets (> 100K rows) | ✅ Yes — primary advantage |
| High-dimensional data | ✅ Yes |
| Speed is critical | ✅ Yes — fastest among GBM variants |
| Categorical features | ✅ Yes — native support |
| Memory-constrained environment | ✅ Yes |
| Small dataset (< 1K rows) | ⚠️ Caution — may overfit |
| Maximum raw accuracy on small data | ⚠️ XGBoost may be comparable |

---

## 🧮 Three Key Innovations

---

### 1️⃣ Histogram-Based Split Finding

```
Standard approach (exact greedy):
  Sort all feature values → scan for best split → O(n log n) per feature

LightGBM histogram approach:
  Bin each feature into k buckets (default: k=255)
  Compute aggregated statistics (sum of gradients/hessians) per bin
  Find best split by scanning k bins instead of n data points

Example (feature "Age", n=100,000 rows):
  Exact: scan 100,000 values per feature per split
  Hist:  scan 255 bins per feature per split → ~400× faster

Trade-off:
  ✅ Much faster training
  ✅ Lower memory (store bins, not raw values)
  ⚠️ Very slight accuracy loss from binning (usually negligible)
```

---

### 2️⃣ GOSS — Gradient-based One-Side Sampling

```
Key insight: Data points with LARGE gradients contribute MORE to learning.
  Large gradient → model is wrong on this point → needs attention
  Small gradient → model is already correct → less informative

GOSS Algorithm:
  1. Sort all instances by gradient magnitude
  2. Keep top a% instances with largest gradients (retain important ones)
  3. Randomly sample b% from the remaining instances (smaller gradients)
  4. Amplify the sampled small-gradient instances by factor (1-a)/b
     to compensate for their underrepresentation

Result:
  Trains on FEWER instances without significantly losing information
  → Faster training while maintaining accuracy

  With a=20%, b=10%:
    Use top 20% (large gradient) + random 10% of remaining
    Total: 20% + 80%×10% = 28% of data → 3.6× fewer instances
```

---

### 3️⃣ EFB — Exclusive Feature Bundling

```
Key insight: In high-dimensional sparse data, many features are mutually exclusive
  (they rarely have non-zero values simultaneously)

EFB Algorithm:
  Detect mutually exclusive feature groups
  Bundle each group into a single feature
  → Reduce #features without losing information

Example:
  One-hot encoded categorical variable: [1,0,0], [0,1,0], [0,0,1]
  → These 3 features are always exclusive (only one is non-zero)
  → Bundle into 1 feature with values 0, 1, 2

Result:
  For text data with 100,000 features: might reduce to 10,000 bundles
  → 10× reduction in feature dimension
```

---

## 🌿 Leaf-Wise vs Level-Wise Tree Growth

The most distinctive structural difference between LightGBM and XGBoost:

```
Level-wise (XGBoost, sklearn GBM):         Leaf-wise (LightGBM):
  Grow tree one full level at a time          Always split the leaf with max gain

Level 1:    [Root]                            [Root]
           /       \                          /     \
Level 2: [A]      [B]                       [A]    [B]
         / \      / \                        |     / \
Level 3:[C][D]  [E][F]                      [C]  [E] [F]
                                                  |
                                                 [G]

Level-wise: balanced, symmetric tree          Leaf-wise: asymmetric, deeper

Leaf-wise advantage:
  ✅ Finds the BEST split globally at each step → lower loss
  ✅ Can model complex patterns with fewer total leaves

Leaf-wise risk:
  ❌ Can over-grow → overfit on small datasets
  Fix: Use num_leaves parameter to limit total leaves
       (NOT max_depth — which is less important in LightGBM)
```

---

## 🎛️ Key Hyperparameters

### Core Parameters

```
n_estimators      : Number of boosting rounds
                    Typical: 100–2000 (use early stopping)

learning_rate     : Shrinkage factor (eta)
                    Typical: 0.01–0.1 (lower = better + more trees)

num_leaves        : MAX IMPORTANT PARAMETER for LightGBM
                    Controls model complexity (like max_depth in XGBoost)
                    Typical: 20–300 (default: 31)
                    Rule: num_leaves < 2^max_depth

max_depth         : Max depth (less critical than num_leaves)
                    Set to -1 (unlimited) and control via num_leaves

min_data_in_leaf  : Min samples per leaf
                    Larger = more regularization = prevents overfitting
                    Typical: 20–100 on large datasets
```

### Sampling Parameters

```
subsample (bagging_fraction): Row subsampling per tree
                              Typical: 0.7–1.0

colsample_bytree (feature_fraction): Feature subsampling per tree
                              Typical: 0.7–1.0

bagging_freq     : Frequency of bagging (0 = no bagging)
                   Set > 0 to enable row subsampling
```

### Regularization Parameters

```
reg_alpha  (lambda_l1): L1 regularization on leaf weights
                         Typical: 0–1.0

reg_lambda (lambda_l2): L2 regularization on leaf weights
                         Typical: 0–1.0

min_gain_to_split      : Minimum gain to perform a split (like gamma in XGBoost)
                         Typical: 0–1.0
```

### Categorical Feature Parameters

```
categorical_feature: Column indices or names of categorical features
                     LightGBM handles categoricals natively without encoding!

max_cat_threshold  : Max number of categories for optimal split
min_data_per_group : Min data per categorical group
```

---

## 🐱 Native Categorical Support

```
One-hot encoding limitations:
  High cardinality → many binary columns → slow + memory-heavy
  Sparsity → EFB helps, but bundling overhead

LightGBM's approach:
  Uses an optimal partitioning algorithm for categorical features
  → Groups categories into left/right branches without one-hot encoding
  → Handles high-cardinality directly

Usage:
  lgb.Dataset(X_train, label=y_train, categorical_feature=['col_name'])
  # OR in sklearn API:
  LGBMClassifier(categorical_features=['col_name'])
```

---

## 📊 LightGBM API — Two Interfaces

### 1. sklearn API
```python
from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    min_child_samples=20,
    random_state=42,
    n_jobs=-1,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)
```

### 2. Native API
```python
import lightgbm as lgb

dtrain = lgb.Dataset(X_train, label=y_train)
dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)

params = {
    'objective'   : 'binary',
    'metric'      : 'binary_logloss',
    'num_leaves'  : 63,
    'learning_rate': 0.05,
    'subsample'   : 0.8,
    'verbose'     : -1,
}
model = lgb.train(
    params, dtrain,
    num_boost_round=1000,
    valid_sets=[dtrain, dval],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)
```

---

## 🆚 LightGBM vs XGBoost vs CatBoost

| Aspect | LightGBM | XGBoost | CatBoost |
|--------|:--------:|:-------:|:--------:|
| Training Speed | ✅ Fastest | ✅ Fast | ✅ Fast |
| Memory Usage | ✅ Lowest | ⚠️ Medium | ⚠️ Medium |
| Accuracy | ✅ Excellent | ✅ Excellent | ✅ Excellent |
| Categorical | ✅ Native | ❌ Manual | ✅ Best |
| Large datasets | ✅ Best | ✅ Good | ✅ Good |
| Small datasets | ⚠️ May overfit | ✅ Good | ✅ Best |
| GPU Support | ✅ Yes | ✅ Yes | ✅ Yes |
| Missing Values | ✅ Yes | ✅ Yes | ✅ Yes |
| Parameter Tuning | ⚠️ More params | ⚠️ Medium | ✅ Fewer params |

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| num_leaves too large | Overfitting on small datasets | Keep num_leaves ≤ 2^(max_depth−1) |
| Not using early stopping | Overfitting | Always set early_stopping |
| min_data_in_leaf too small | Overfitting noisy data | Set 20–100 for large datasets |
| Using max_depth instead of num_leaves | Not the primary control | Focus on num_leaves |
| Ignoring feature_fraction | All features per split | Set 0.7–0.9 for regularization |
| verbose=-1 not set | Floods console with logs | Always set verbose=-1 |

---

## 🔗 Related Topics

- `Gradient_Boosting` — The algorithm LightGBM implements
- `XGBoost` — Alternative GBDT with different trade-offs
- `CatBoost` — Best native categorical feature handling
- `07_Hyperparameter_Tuning` — Optuna / Bayesian optimization for LightGBM
- `08_Ensemble_Learning` — Stacking LightGBM with other models

---

## 📚 References

- LightGBM Docs: [https://lightgbm.readthedocs.io](https://lightgbm.readthedocs.io)
- Original LightGBM Paper (Ke et al., 2017): [https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
- LightGBM sklearn API: [https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html)
- Parameter Tuning Guide: [https://lightgbm.readthedocs.io/en/stable/Parameters_Tuning.html](https://lightgbm.readthedocs.io/en/stable/Parameters_Tuning.html)
