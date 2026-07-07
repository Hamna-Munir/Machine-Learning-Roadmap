# 📘 XGBoost — Theory

---

## 📌 What is XGBoost?

XGBoost (**eXtreme Gradient Boosting**) is an **optimized, regularized  
gradient boosting framework** developed by Tianqi Chen. It extends  
standard GBM with second-order gradient optimization, built-in L1+L2  
regularization, and hardware-aware computation — making it faster,  
more accurate, and more robust than vanilla Gradient Boosting.

```
Standard GBM:  Uses first-order gradient (pseudo-residuals)
XGBoost:       Uses first AND second-order gradient (Newton's method)
               + L1 + L2 regularization on leaf weights
               + Column (feature) subsampling per tree AND per level
               + Missing value handling built-in
               + Parallel tree construction
```

> 💡 "XGBoost won more Kaggle competitions between 2015–2017  
>      than any other single algorithm — and it's still a top-tier choice  
>      for structured/tabular data."

---

## 🔍 When to Use XGBoost?

| Condition | Use XGBoost? |
|-----------|:-----------:|
| Structured / tabular data | ✅ Yes — primary strength |
| Maximum accuracy on tabular data | ✅ Yes |
| Need built-in regularization | ✅ Yes (L1 + L2) |
| Handle missing values automatically | ✅ Yes |
| Medium-large datasets (1K–1M rows) | ✅ Yes |
| Very large datasets (> 10M rows) | ⚠️ Prefer LightGBM |
| Image/text/sequence data | ❌ No → Use deep learning |
| Need interpretability | ❌ No → Use Decision Tree |

---

## 🧮 XGBoost Objective Function

XGBoost minimizes a regularized objective:

```
Obj(Θ) = Σᵢ L(yᵢ, ŷᵢ) + Σₖ Ω(fₖ)
          └── Loss term ──┘  └── Regularization ──┘

Where:
  L(yᵢ, ŷᵢ) = Loss function (log loss for classification,
                               squared error for regression)

  Ω(f)       = γT + (1/2)λ||w||²  + α||w||₁
               └─ L0  ─┘ └── L2 ──┘  └── L1 ──┘

  T  = number of leaves in the tree
  w  = leaf weight vector
  γ  = minimum loss reduction to split (complexity penalty on tree size)
  λ  = L2 regularization on leaf weights (Ridge-like)
  α  = L1 regularization on leaf weights (Lasso-like)
```

---

## 🔧 Second-Order (Newton) Optimization

Standard GBM uses only the **first-order gradient** (like gradient descent).  
XGBoost uses **second-order Taylor expansion** of the loss:

```
L(yᵢ, ŷᵢ⁽ᵐ⁾) ≈ L(yᵢ, ŷᵢ⁽ᵐ⁻¹⁾) + gᵢfₘ(xᵢ) + (1/2)hᵢfₘ(xᵢ)²

Where:
  gᵢ = ∂L/∂ŷᵢ        (first-order gradient — same as GBM)
  hᵢ = ∂²L/∂ŷᵢ²       (second-order gradient / Hessian)

Optimal leaf weight for leaf j:
  w*ⱼ = − (Σᵢ∈j gᵢ) / (Σᵢ∈j hᵢ + λ)

Gain from splitting node into left (L) and right (R):
  Gain = (1/2) [ GL²/(HL+λ) + GR²/(HR+λ) − (GL+GR)²/(HL+HR+λ) ] − γ

Benefits of second-order optimization:
  ✅ Faster convergence than first-order methods
  ✅ More accurate step sizes (curvature information)
  ✅ Mathematically rigorous — derived from Newton's method
```

---

## 🎛️ Key Hyperparameters

### Core Boosting Parameters

```
n_estimators    : Number of trees (boosting rounds)
                  Typical: 100–1000, use early stopping to find optimal

learning_rate   : Shrinkage factor η (also called eta)
                  Typical: 0.01–0.3 (lower = better + more trees needed)

max_depth       : Maximum tree depth
                  Typical: 3–10 (deeper = more complex interactions)

subsample       : Row subsampling fraction per tree
                  Typical: 0.5–1.0 (reduces variance)

colsample_bytree: Feature subsampling fraction per tree
                  Typical: 0.5–1.0

colsample_bylevel: Feature subsampling per tree level
                  Finer control than colsample_bytree

colsample_bynode: Feature subsampling per split node
                  Most fine-grained control
```

### Regularization Parameters

```
gamma (min_child_weight):
  Minimum loss reduction required to split a node
  Higher gamma → more conservative splitting → simpler trees
  Typical: 0–5

reg_alpha (α): L1 regularization on leaf weights
  Promotes sparsity — zeroes out small leaf weights
  Typical: 0, 0.1, 1.0

reg_lambda (λ): L2 regularization on leaf weights
  Shrinks leaf weights smoothly
  Default: 1.0  (always on)

min_child_weight:
  Minimum sum of instance weights (hessian) in a child
  Higher = more conservative = prevents overfitting on small nodes
  Typical: 1–10
```

### Speed & Hardware Parameters

```
tree_method : 'hist' (histogram-based, fast), 'exact', 'approx'
              Use 'hist' for large datasets

n_jobs      : Number of parallel threads (-1 = all cores)

device      : 'cpu' or 'cuda' (GPU acceleration)
```

---

## 🌟 Key Innovations Over Standard GBM

### 1. Regularization (Built-in)
```
Standard GBM: No explicit regularization on tree structure
XGBoost:      L1 (α) + L2 (λ) + γ (complexity) built into the objective
→ Reduces overfitting without needing separate tuning of tree depth alone
```

### 2. Missing Value Handling
```
Standard GBM: Requires imputation before training
XGBoost:      Learns the optimal direction (left or right branch)
              for missing values during training
→ No preprocessing needed for NaN values ✅
```

### 3. Column Subsampling (Three Levels)
```
colsample_bytree  → Random feature subset per tree (like RF)
colsample_bylevel → Random feature subset per tree level
colsample_bynode  → Random feature subset per split

→ More granular variance reduction than standard GBM
```

### 4. Histogram-Based Splits (tree_method='hist')
```
Instead of exact split finding (O(n×d)):
  → Bin continuous features into k buckets (~256)
  → Find best split within bins: O(k×d) — much faster

→ Same approach as LightGBM, enables large-scale training
```

### 5. Early Stopping
```
xgb.train(..., early_stopping_rounds=50)
→ Stop training when validation metric doesn't improve for 50 rounds
→ Prevents overfitting and saves compute time
→ Best practice: always use with a validation set
```

---

## 📊 XGBoost API — Two Interfaces

### 1. sklearn API (recommended for consistency)
```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    eval_metric='logloss',
    early_stopping_rounds=50,
)
model.fit(X_train, y_train,
           eval_set=[(X_val, y_val)],
           verbose=False)
```

### 2. Native API (more control)
```python
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val,   label=y_val)

params = {
    'objective'  : 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth'  : 5,
    'eta'        : 0.1,
    'subsample'  : 0.8,
}
model = xgb.train(
    params, dtrain,
    num_boost_round=500,
    evals=[(dtrain,'train'), (dval,'val')],
    early_stopping_rounds=50,
    verbose_eval=False
)
```

---

## 🎯 Objective Functions

| Task | objective | Description |
|------|-----------|-------------|
| Binary classification | `binary:logistic` | Log loss, outputs probability |
| Multiclass classification | `multi:softmax` | Softmax, outputs class |
| Multiclass (with proba) | `multi:softprob` | Softmax, outputs probabilities |
| Regression (L2) | `reg:squarederror` | Mean Squared Error |
| Regression (L1) | `reg:absoluteerror` | Mean Absolute Error |
| Ranking | `rank:pairwise` | Pairwise ranking |

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| No early stopping | Overfits at large n_estimators | Always use early_stopping_rounds |
| Default max_depth=6 too large | High variance | Start with max_depth=3–5 |
| Not setting random_state | Non-reproducible results | Always set seed |
| Not using colsample_bytree | All features every split | Set 0.5–0.8 |
| Ignoring scale_pos_weight | Imbalanced classes | Set = n_negative/n_positive |
| Using CPU for large data | Slow training | Use tree_method='hist' |
| No eval_metric set | Can't monitor validation loss | Set eval_metric explicitly |

---

## 🆚 XGBoost vs GBM vs LightGBM vs CatBoost

| Aspect | GBM (sklearn) | XGBoost | LightGBM | CatBoost |
|--------|:-------------:|:-------:|:--------:|:--------:|
| Speed | ❌ Slow | ✅ Fast | ✅ Very Fast | ✅ Fast |
| Regularization | ⚠️ Limited | ✅ L1+L2+γ | ✅ L1+L2 | ✅ Built-in |
| Missing values | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| Categoricals | ❌ No | ❌ No | ✅ Yes | ✅ Yes |
| Memory | ⚠️ High | ⚠️ Medium | ✅ Low | ✅ Medium |
| Accuracy | ✅ Good | ✅ Excellent | ✅ Excellent | ✅ Excellent |
| GPU Support | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |

---

## 🔗 Related Topics

- `Gradient_Boosting` — The algorithm XGBoost extends
- `LightGBM` — Faster alternative, leaf-wise growth
- `CatBoost` — Native categorical support
- `08_Ensemble_Learning` — Boosting theory
- `07_Hyperparameter_Tuning` — GridSearchCV + Bayesian optimization for XGBoost

---

## 📚 References

- XGBoost Docs: [https://xgboost.readthedocs.io](https://xgboost.readthedocs.io)
- Original XGBoost Paper (Chen & Guestrin, 2016): [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)
- XGBoost sklearn API: [https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html](https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html)
- An Introduction to Statistical Learning — Chapter 8.2
