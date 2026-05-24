# 📘 Lasso Regression — Theory

---

## 📌 What is Lasso Regression?

Lasso Regression (**Least Absolute Shrinkage and Selection Operator**) is a regularized  
version of Linear Regression that adds an **L1 penalty** — the sum of **absolute values**  
of coefficients — to the loss function.

Unlike Ridge, Lasso can shrink coefficients to **exactly zero**, performing automatic  
**feature selection** as a byproduct of regularization.

```
Linear Regression Loss:  L = Σ(yᵢ − ŷᵢ)²
Lasso Regression Loss:   L = Σ(yᵢ − ŷᵢ)² + λ × Σ|βⱼ|
                                               └─── L1 penalty
```

> 💡 "Lasso is not just a regularizer — it is a **feature selector**.  
>      It produces sparse models by eliminating irrelevant features completely."

---

## 🔍 When to Use Lasso?

| Condition | Use Lasso? |
|-----------|:---------:|
| Many features, only a few are truly relevant | ✅ Yes — primary use case |
| Automatic feature selection needed | ✅ Yes |
| Sparse solution desired | ✅ Yes |
| Features are highly correlated | ⚠️ Caution — Lasso picks one arbitrarily |
| All features are likely relevant | ❌ No → Use Ridge |
| Need a closed-form solution | ❌ No → Ridge has one, Lasso doesn't |
| Target is categorical | ❌ No → Use Logistic Regression |

---

## 🧮 The Model

### Loss Function (Objective)

```
Lasso Loss = RSS + λ × L1 Penalty

         n                    p
L(β) =  Σ (yᵢ − ŷᵢ)²  +  λ × Σ |βⱼ|
        i=1                  j=1

Where:
  RSS   = Residual Sum of Squares
  λ     = regularization strength (alpha in sklearn)
  |βⱼ|  = absolute value of each coefficient
  p     = number of features
  β₀    = intercept (NOT penalized)
```

### Effect of Alpha

```
alpha = 0     →  No regularization (= Linear Regression)
alpha → ∞    →  All coefficients → 0 (only intercept remains)
Optimal alpha minimizes cross-validation error

Small alpha:  Few zero coefficients, model close to OLS
Large alpha:  Many zero coefficients, very sparse model
```

---

## 🔧 Optimization — Coordinate Descent

Unlike Ridge, Lasso has **no closed-form analytical solution** — the L1 norm is  
not differentiable at zero. It is solved using **coordinate descent**:

```
Coordinate Descent Algorithm:
  1. Initialize all β to 0 (or OLS solution)
  2. For each feature j:
       a. Compute partial residual rⱼ = y − Xβ + βⱼxⱼ
       b. Update βⱼ using the soft-thresholding operator:

              ┌  rⱼᵀxⱼ/n − λ   if  rⱼᵀxⱼ/n >  λ
    β̂ⱼ =   {  0               if  |rⱼᵀxⱼ/n| ≤ λ
              └  rⱼᵀxⱼ/n + λ   if  rⱼᵀxⱼ/n < −λ

  3. Repeat until convergence (all βⱼ stop changing)
```

**Soft-Thresholding Function:**
```
         β
         │      ● /  ← Lasso solution
         │     /
         │    /
─────────┼───────────── β_OLS
         │   /
         │  / λ gap
         │ /
         ●

Values within ±λ of zero are set to exactly 0 — this is what creates sparsity
```

---

## 📐 Geometric Interpretation

Lasso constrains coefficients to lie within an **L1 ball (diamond shape)**:

```
            β₂
             │
             │    ●  OLS solution (unconstrained)
             │   /
        ♦    │  /
       /  \  │ /
      /    \ │/
─────◆──────◆──────────────── β₁
      \    /│\
       \  / │ \
        ♦   │  \
             │

L1 constraint: |β₁| + |β₂| ≤ t  (diamond)
Loss contours: ellipses centered at OLS solution

Lasso solution = point where loss contour first touches the L1 diamond
→ Touches at a CORNER of the diamond → one (or more) coefficient = 0 ✅

Compare with Ridge (circle): touches at a curved edge → never exactly 0
```

---

## 🎛️ Alpha (λ) — The Regularization Parameter

```
sklearn parameter: alpha  (equivalent to λ in formulas)

alpha = 0.0    →  No regularization (= OLS Linear Regression)
alpha = 0.001  →  Very mild: almost all features retained
alpha = 0.1    →  Moderate: some irrelevant features zeroed out
alpha = 1.0    →  Strong: significant feature elimination
alpha = 10.0   →  Very strong: most features zeroed out

Selection strategy:
  → Always search on a log scale: [0.0001, 0.001, 0.01, 0.1, 1, 10]
  → Use LassoCV or GridSearchCV for automatic selection
```

---

## 📊 Coefficient Behavior vs Alpha

```
Feature     | OLS    | α=0.01 | α=0.1  | α=1.0  | α=10.0
──────────────────────────────────────────────────────────────
Feature_1   |  2.50  |  2.48  |  2.31  |  1.20  |  0.00  ← zeroed
Feature_2   |  1.80  |  1.79  |  1.62  |  0.60  |  0.00  ← zeroed
Feature_3   |  0.05  |  0.04  |  0.00  |  0.00  |  0.00  ← zeroed early
Feature_4   | -3.20  | -3.18  | -3.05  | -2.10  | -0.80
Feature_5   |  0.80  |  0.79  |  0.65  |  0.00  |  0.00  ← zeroed

Observations:
  - Small coefficients (Feature_3) are zeroed out first
  - Larger coefficients survive longer
  - Eventually all reach 0 as alpha → ∞
```

---

## 🆚 Lasso vs Ridge vs ElasticNet

| Aspect | Lasso (L1) | Ridge (L2) | ElasticNet (L1+L2) |
|--------|:----------:|:----------:|:-----------------:|
| Penalty | Σ\|βⱼ\| | Σβⱼ² | α×Σ\|βⱼ\| + (1−α)×Σβⱼ² |
| Feature Selection | ✅ Yes — zeros out | ❌ No | ✅ Partial |
| Correlated Features | ⚠️ Picks one | ✅ Distributes | ✅ Groups together |
| Solution | Coordinate descent | Analytical | Coordinate descent |
| Sparsity | ✅ Sparse | ❌ Dense | ✅ Moderate |
| Best For | Many irrelevant features | Multicollinearity | Both issues |

---

## 📈 Regularization Path — How Coefficients Are Zeroed

```
Coefficient value
│  β₁ ─────────────────────────────────────── ●
│        β₄ ──────────────────────── ●
│              β₂ ─────────── ●
│                   β₅ ─── ●
│                        β₃ ●
└─────────────────────────────────────────── alpha (log scale)
  0   0.001  0.01   0.1    1    10   100

Features exit (→0) one by one as alpha increases
Last features remaining = most important predictors
```

This path is called the **Lasso path** — it reveals feature importance ranking.

---

## 🔧 sklearn Implementation

```python
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Option 1: Lasso with fixed alpha
model = Lasso(alpha=0.1, max_iter=10000)

# Option 2: LassoCV — built-in cross-validation for alpha selection
model = LassoCV(alphas=None, cv=5, max_iter=10000)
model.fit(X_train_scaled, y_train)
print(f'Best alpha: {model.alpha_}')

# Option 3: Pipeline (recommended)
pipe = Pipeline([
    ('scaler', StandardScaler()),    # MUST scale before Lasso
    ('model',  Lasso(alpha=0.1, max_iter=10000))
])
```

**Why must you scale before Lasso?**
```
L1 penalty:  λ × (|β₁| + |β₂| + ... + |βₚ|)

Without scaling:
  Feature with range [0, 1000] → small β₁ → weak penalization
  Feature with range [0, 1]    → large β₂ → over-penalized ❌

With scaling (StandardScaler):
  All features in same range → equal penalization of all β ✅
  Comparable coefficient magnitudes
```

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Not scaling before Lasso | Unequal penalization | Always StandardScale first |
| Using default max_iter=1000 | May not converge | Set max_iter=10000+ |
| Too large alpha | All features zeroed out | Start small, use LassoCV |
| Correlated features | Arbitrary selection of one | Use ElasticNet or Ridge instead |
| Treating as Ridge | Lasso can zero features | Check which coefficients are 0 |
| Not checking convergence warning | Model didn't fully converge | Increase max_iter or scale features |

---

## 🔗 Related Topics

- `Linear_Regression` — Foundation that Lasso regularizes
- `Ridge_Regression` — L2 regularization — no feature selection
- `ElasticNet` — Combines L1 + L2 to handle Lasso's weaknesses
- `Polynomial_Regression` — Apply Lasso for sparse polynomial selection
- `06_Feature_Selection` — Lasso coefficients as feature importance
- `07_Hyperparameter_Tuning` — LassoCV and GridSearchCV for alpha

---

## 📚 References

- Scikit-learn `Lasso`: [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
- Scikit-learn `LassoCV`: [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html)
- Original Lasso Paper (Tibshirani, 1996): [https://www.jstor.org/stable/2346178](https://www.jstor.org/stable/2346178)
- The Elements of Statistical Learning — Chapter 3.4
- An Introduction to Statistical Learning — Chapter 6.2
