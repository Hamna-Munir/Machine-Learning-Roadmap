# 📘 CatBoost — Theory

---

## 📌 What is CatBoost?

CatBoost (**Categorical Boosting**) is a **gradient boosting framework**  
developed by Yandex that natively handles **categorical features** without  
requiring manual encoding — while also delivering state-of-the-art accuracy  
on tabular data with minimal hyperparameter tuning.

```
Standard GBM / XGBoost:
  Categorical features → Manual encoding required (one-hot, label, target enc.)
  → Risk of target leakage with naive target encoding
  → High-cardinality categories → high memory usage

CatBoost:
  Categorical features → Passed directly to the model
  → Uses Ordered Target Statistics (leak-free encoding)
  → Handles high-cardinality categories efficiently
  → Ordered boosting prevents prediction shift
```

> 💡 "CatBoost is the 'plug-and-play' gradient booster —  
>      minimal preprocessing, state-of-the-art accuracy,  
>      especially when your data has many categorical columns."

---

## 🔍 When to Use CatBoost?

| Condition | Use CatBoost? |
|-----------|:------------:|
| Dataset has many categorical features | ✅ Yes — primary strength |
| Want to avoid manual encoding | ✅ Yes |
| Minimal preprocessing pipeline desired | ✅ Yes |
| Small to medium datasets | ✅ Yes — doesn't overfit easily |
| Symmetric / explainable trees needed | ✅ Yes |
| Very large datasets (> 10M rows) | ⚠️ LightGBM may be faster |
| Pure numerical features only | ⚠️ XGBoost / LightGBM may be comparable |

---

## 🧮 Three Key Innovations

---

### 1️⃣ Ordered Target Statistics (Leak-Free Categorical Encoding)

Standard target encoding:
```
City → TargetMean(City)

Problem: Uses target of CURRENT sample to encode itself → data leakage!
```

CatBoost's solution — Ordered TS:
```
For each sample i with categorical value c:
  Encode c using statistics from ONLY the samples that appeared
  BEFORE sample i in a random permutation of the data

TS(xᵢ, c) = (count of {j < i : xⱼ = c AND yⱼ = 1} + prior) /
             (count of {j < i : xⱼ = c} + 1)

Where prior = overall target mean × prior_weight

Benefits:
  ✅ No target leakage — future samples never included
  ✅ Handles high cardinality naturally
  ✅ Smoothed with prior to avoid overfitting on rare categories
```

---

### 2️⃣ Ordered Boosting (Oblivious Gradient Estimation)

Standard GBM problem — **Prediction Shift**:
```
Standard GBM builds residuals on the SAME data used to train the tree
→ Tree fitting is biased (model was built to explain these exact errors)
→ Predictions are shifted toward the training errors
→ Leads to overfitting, especially on small datasets
```

CatBoost solution — Ordered Boosting:
```
Maintain a separate model for each sample, trained WITHOUT that sample
(similar in concept to leave-one-out cross-validation)

Implementation (efficient approximation):
  Random permutation σ of training data
  For sample i:
    Model Mᵢ is built using only samples {σ(1), ..., σ(i-1)}
    Gradient of sample i computed using Mᵢ (no leakage)

Benefits:
  ✅ Unbiased gradient estimates
  ✅ Reduced prediction shift → better generalization
  ✅ Particularly effective on small datasets
```

---

### 3️⃣ Symmetric (Oblivious) Trees

CatBoost uses **symmetric decision trees** — where every node at the same  
depth uses the SAME split condition:

```
Standard asymmetric tree:           CatBoost symmetric tree:
      [Age < 30?]                        [Age < 30?]
      /          \                       /           \
[Income < 50K?] [Score > 7?]    [Income < 50K?] [Income < 50K?]
  /    \           /    \           /    \          /    \
[C0] [C1]      [C2]  [C3]      [C0]  [C1]      [C2]  [C3]

Asymmetric: each node can split on    Symmetric: all nodes at depth d
different feature/threshold           use the same feature/threshold

Benefits of symmetric trees:
  ✅ Fast prediction: O(depth) evaluation using lookup tables
  ✅ Less overfitting: fewer parameters to fit
  ✅ Regular structure enables SHAP explanations
  ✅ Works well when depth is shallow (typical: 6-10)
```

---

## 🎛️ Key Hyperparameters

### Core Parameters

```
iterations (n_estimators): Number of trees
                            Typical: 100–3000 (use early stopping)

learning_rate             : Shrinkage factor
                            Default: auto-computed based on dataset size
                            Typical: 0.03–0.3

depth                     : Tree depth (symmetric tree)
                            Typical: 6–10 (shallower than XGBoost/LightGBM
                            because symmetric trees are less complex per level)

l2_leaf_reg               : L2 regularization on leaf values
                            Default: 3.0
                            Typical: 1–10
```

### Categorical Feature Parameters

```
cat_features              : List of categorical feature indices or names
                            CatBoost handles these natively

one_hot_max_size          : Max cardinality for one-hot encoding
                            Categories above this use target statistics
                            Default: 2 (binary → one-hot, rest → TS)

model_size_reg            : Regularizes the model size
                            Higher → smaller model → less overfitting
```

### Sampling Parameters

```
subsample (bootstrap_type): Row subsampling
                             'Bernoulli': random fraction
                             'MVS': Minimum Variance Sampling (default for GPU)

rsm                        : Feature subsampling (like colsample_bytree)
                             Typical: 0.8–1.0
```

### Speed Parameters

```
task_type   : 'CPU' (default) or 'GPU'
thread_count: Number of CPU threads (-1 = all)
verbose     : 0 to suppress training output
```

---

## 📊 CatBoost API

### Python API
```python
from catboost import CatBoostClassifier, CatBoostRegressor

# Identify categorical columns
cat_cols = ['City', 'Education', 'Gender']

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    cat_features=cat_cols,     # pass raw categorical columns directly!
    eval_metric='AUC',
    early_stopping_rounds=50,
    verbose=0,
    random_seed=42,
)

model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    verbose=False
)
```

---

## 🎯 Supported Objectives

| Task | eval_metric | loss_function |
|------|:-----------:|:-------------:|
| Binary classification | 'AUC', 'F1', 'Logloss' | 'Logloss' (default) |
| Multiclass classification | 'Accuracy', 'MultiClass' | 'MultiClass' |
| Regression | 'RMSE', 'MAE', 'R2' | 'RMSE' (default) |
| Ranking | 'NDCG', 'MAP' | 'YetiRank' |
| Quantile regression | 'Quantile' | 'Quantile' |

---

## 🌟 SHAP Values — Built-in Explainability

CatBoost has built-in SHAP value computation (faster than shap library):

```python
# Get SHAP values
shap_values = model.get_feature_importance(
    Pool(X_test, cat_features=cat_cols),
    type='ShapValues'
)
# shap_values shape: (n_samples, n_features + 1)
# Last column = expected value (bias)

# Or use standard shap library
import shap
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
```

---

## 📈 Feature Importance Types

```
1. PredictionValuesChange (default):
   How much predictions change when feature is varied
   Fast to compute, good for ranking

2. LossFunctionChange:
   How much loss changes when feature is removed
   More accurate but slower

3. ShapValues:
   Additive attribution — SHAP framework
   Most theoretically sound

4. Interaction:
   Feature interaction strength (pairwise)
```

---

## 🆚 CatBoost vs XGBoost vs LightGBM

| Aspect | CatBoost | XGBoost | LightGBM |
|--------|:--------:|:-------:|:--------:|
| Categorical Features | ✅ Best | ❌ Manual | ✅ Good |
| Small Datasets | ✅ Best | ✅ Good | ⚠️ May overfit |
| Large Datasets | ✅ Good | ✅ Good | ✅ Best |
| Training Speed | ✅ Fast | ✅ Fast | ✅ Fastest |
| Tuning Required | ✅ Minimal | ⚠️ Medium | ⚠️ Medium |
| Prediction Speed | ✅ Fastest | ✅ Fast | ✅ Fast |
| GPU Support | ✅ Yes | ✅ Yes | ✅ Yes |
| Built-in SHAP | ✅ Yes | ⚠️ External | ⚠️ External |
| Out-of-box accuracy | ✅ Excellent | ✅ Excellent | ✅ Excellent |

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Encoding categoricals before CatBoost | Loses CatBoost advantage | Pass raw strings directly |
| Not setting verbose=0 | Floods console every iteration | Always set verbose=0 |
| Not specifying cat_features | Treated as numerical | Always pass cat_features list |
| Default depth=6 on complex data | May underfit | Try depth=8–10 |
| Not using early stopping | Overfits at large iterations | Always set early_stopping_rounds |
| Using OHE for high-cardinality | Memory explosion | Let CatBoost handle natively |

---

## 🔗 Related Topics

- `Gradient_Boosting` — The base algorithm CatBoost implements
- `XGBoost` — Alternative GBDT with different trade-offs
- `LightGBM` — Fastest GBDT for large datasets
- `08_Ensemble_Learning` — Stacking CatBoost with other models
- `07_Hyperparameter_Tuning` — Optuna / Bayesian optimization for CatBoost

---

## 📚 References

- CatBoost Docs: [https://catboost.ai/docs](https://catboost.ai/docs)
- CatBoost Python API: [https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html](https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html)
- Original CatBoost Paper (Prokhorenkova et al., 2018): [https://arxiv.org/abs/1706.09516](https://arxiv.org/abs/1706.09516)
- CatBoost GitHub: [https://github.com/catboost/catboost](https://github.com/catboost/catboost)
