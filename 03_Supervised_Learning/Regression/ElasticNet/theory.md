# 📘 ElasticNet Regression — Theory

---

## 📌 What is ElasticNet?

ElasticNet is a regularized regression model that **combines both L1 (Lasso) and L2 (Ridge)**  
penalties in its loss function — inheriting the best properties of both:

- From **Lasso (L1)**: Feature selection — some coefficients become exactly zero
- From **Ridge (L2)**: Stability — handles correlated features gracefully

```
Linear Regression:  L = RSS
Ridge Regression:   L = RSS + λ × Σβⱼ²
Lasso Regression:   L = RSS + λ × Σ|βⱼ|
ElasticNet:         L = RSS + λ₁ × Σ|βⱼ| + λ₂ × Σβⱼ²
                              └── L1 (Lasso)   └── L2 (Ridge)
```

> 💡 "ElasticNet is the compromise — it selects features like Lasso  
>     but handles correlated feature groups like Ridge."

---

## 🔍 Why ElasticNet?

ElasticNet was introduced to address the two main weaknesses of pure Lasso:

### Lasso's Weaknesses That ElasticNet Fixes

```
Problem 1 — Correlated Features:
  Lasso   → arbitrarily picks ONE feature from a correlated group, drops the rest
  ElasticNet → keeps the GROUP together, distributing coefficient weight ✅

Problem 2 — n < p (more features than samples):
  Lasso   → can select at most n features (limited by sample count)
  ElasticNet → can select more than n features ✅

Problem 3 — Instability on near-identical features:
  Lasso   → coefficient can flip sign erratically
  ElasticNet → stable, consistent selection ✅
```

---

## 🧮 The Model

### Loss Function

```
ElasticNet Loss = RSS + Penalty

L(β) = Σ(yᵢ − ŷᵢ)² + α × [l1_ratio × Σ|βⱼ| + (1 − l1_ratio) × Σβⱼ²]
                         └────────────────────────────────────────────┘
                                       Combined Penalty

Where:
  α          = overall regularization strength (alpha in sklearn)
  l1_ratio   = mixing parameter between L1 and L2 (0 to 1)
  l1_ratio=1 → pure Lasso
  l1_ratio=0 → pure Ridge
  l1_ratio=0.5 → equal mix (default in many implementations)
```

### sklearn Parameterization

```python
ElasticNet(alpha=1.0, l1_ratio=0.5)

alpha    : Overall regularization strength
           → Controls total shrinkage (like λ in formulas)

l1_ratio : Balance between L1 and L2
           0.0  → Ridge (no L1)
           0.5  → Equal mix (default)
           1.0  → Lasso (no L2)
           0.1–0.9 → typical useful range
```

---

## 🔧 Optimization

ElasticNet is solved using **coordinate descent** (same as Lasso):

```
For each feature j, the soft-thresholding update becomes:

           ┌  (rⱼᵀxⱼ/n − α×l1_ratio) / (1 + α×(1−l1_ratio))   if > 0
β̂ⱼ =    {  0                                                     if |...| ≤ α×l1_ratio
           └  (rⱼᵀxⱼ/n + α×l1_ratio) / (1 + α×(1−l1_ratio))   if < 0

The L2 term (denominator): 1 + α×(1−l1_ratio)
→ shrinks coefficients like Ridge (groups correlated features)

The L1 term (numerator shift): α×l1_ratio
→ thresholds small coefficients to zero like Lasso
```

---

## 📐 Geometric Interpretation

ElasticNet combines the L1 diamond and L2 circle constraint regions:

```
            β₂
             │
         ╭───╮        ← ElasticNet constraint (rounded diamond)
        ╱     ╲           = between L1 diamond and L2 circle
       │    ●──│── OLS solution
       │       │
────────╲─────╱─────────── β₁
         ╰───╯

L1 diamond:    sharp corners → exact zeros (feature selection)
L2 circle:     smooth edge  → no exact zeros
ElasticNet:    rounded diamond → some exact zeros + grouping ✅

The more L1 (higher l1_ratio) → more corner-like → more zeros
The more L2 (lower l1_ratio)  → more circular   → fewer zeros
```

---

## 🎛️ Two Hyperparameters

### 1. alpha — Overall Regularization Strength

```
alpha = 0.0    →  No regularization (pure OLS)
alpha = 0.01   →  Mild: minimal shrinkage
alpha = 0.1    →  Moderate: some features may be zeroed
alpha = 1.0    →  Strong: significant sparsity
alpha = 100.0  →  Very strong: most features zeroed out

→ Search on log scale: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
```

### 2. l1_ratio — L1 vs L2 Mix

```
l1_ratio = 0.0  →  Pure Ridge   (no selection, full grouping)
l1_ratio = 0.1  →  Mostly Ridge
l1_ratio = 0.5  →  Equal mix    (sklearn default)
l1_ratio = 0.9  →  Mostly Lasso
l1_ratio = 1.0  →  Pure Lasso   (maximum selection, no grouping)

→ Recommended range to search: [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
→ Values near 0 behave like Ridge, near 1 like Lasso
```

---

## 📊 Coefficient Behavior

```
Feature     | OLS    | Ridge  | Lasso  | ElasticNet
──────────────────────────────────────────────────────
Feature_A   |  3.20  |  2.95  |  2.80  |  2.90    ← relevant (kept by all)
Feature_B   |  2.80  |  2.55  |  2.40  |  2.52    ← relevant
Feature_A'  |  3.10  |  2.90  |  0.00  |  1.95    ← correlated with A
              (same as A)      ↑              ↑
                              Lasso drops one  ElasticNet keeps both ✅
Feature_C   | -0.05  | -0.03  |  0.00  |  0.00    ← noise (correctly zeroed)
Feature_D   |  0.02  |  0.01  |  0.00  |  0.00    ← noise (correctly zeroed)

Key insight: ElasticNet keeps correlated feature groups (A and A'),
             while Lasso arbitrarily picks just one.
```

---

## 🆚 ElasticNet vs Ridge vs Lasso

| Aspect | Ridge (L2) | Lasso (L1) | ElasticNet (L1+L2) |
|--------|:----------:|:----------:|:-----------------:|
| Penalty | Σβⱼ² | Σ\|βⱼ\| | l1×Σ\|βⱼ\| + l2×Σβⱼ² |
| Feature Selection | ❌ Never | ✅ Yes | ✅ Yes (partial) |
| Correlated Features | ✅ Groups well | ❌ Drops arbitrarily | ✅ Groups well |
| n < p | ✅ Works | ⚠️ Limited | ✅ Works |
| Hyperparameters | 1 (alpha) | 1 (alpha) | 2 (alpha + l1_ratio) |
| Sparsity | ❌ Dense | ✅ Sparse | ✅ Moderate |
| Best For | Multicollinearity | Many irrelevant features | Both problems |
| Tuning Complexity | Low | Low | Medium |

---

## 🧠 When to Use ElasticNet?

```
Use ElasticNet when:
  ✅ You have BOTH correlated features AND irrelevant features
  ✅ Number of features > number of samples (p >> n)
  ✅ You want feature selection but Ridge instability concerns you
  ✅ Features come in natural groups (genomics, text, multi-collinear tabular)

Use Ridge when:
  → Features are correlated but all likely relevant
  → You don't need sparsity

Use Lasso when:
  → Features are mostly independent
  → You want maximum sparsity
  → You have clear irrelevant features to eliminate
```

---

## 📈 Bias-Variance Tradeoff

```
As alpha increases (for any l1_ratio):
  Bias↑   Variance↓   (model gets simpler)

As l1_ratio increases (for any alpha):
  More sparsity, more feature selection behavior
  Better if features are truly independent
  Riskier if features are correlated
```

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Not scaling before ElasticNet | Unequal penalization | Always StandardScale first |
| Default max_iter too low | Convergence warning | Set max_iter=10000+ |
| Only tuning alpha | l1_ratio equally important | Grid search both simultaneously |
| l1_ratio=1.0 with correlated features | Lasso instability | Use l1_ratio=0.5–0.9 |
| l1_ratio=0.0 | Same as Ridge — loses selection | Keep l1_ratio > 0 for sparsity |
| Comparing without scaling | Coefficients not comparable | Scale before comparing |

---

## 🔧 sklearn Implementation

```python
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Option 1: Fixed hyperparameters
model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)

# Option 2: ElasticNetCV — built-in cross-validation
model = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
    alphas=None,   # auto log-scale
    cv=5,
    max_iter=10000
)
model.fit(X_train_scaled, y_train)
print(f'Best alpha   : {model.alpha_}')
print(f'Best l1_ratio: {model.l1_ratio_}')

# Option 3: Pipeline (recommended)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000))
])
```

---

## 🔗 Related Topics

- `Linear_Regression` — Unregularized baseline
- `Ridge_Regression` — Pure L2 regularization
- `Lasso_Regression` — Pure L1 regularization
- `Polynomial_Regression` — Apply ElasticNet after feature expansion
- `07_Hyperparameter_Tuning` — GridSearchCV for alpha + l1_ratio
- `06_Feature_Selection` — ElasticNet for embedded feature selection

---

## 📚 References

- Scikit-learn `ElasticNet`: [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
- Scikit-learn `ElasticNetCV`: [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html)
- Original ElasticNet Paper (Zou & Hastie, 2005): [https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2005.00503.x](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2005.00503.x)
- The Elements of Statistical Learning — Chapter 3.4
- An Introduction to Statistical Learning — Chapter 6.2
