# 📘 Ridge Regression — Theory

---

## 📌 What is Ridge Regression?

Ridge Regression (also called **L2 Regularization** or **Tikhonov Regularization**) is a  
modified version of Linear Regression that adds a **penalty term** to the loss function  
proportional to the **sum of squared coefficients** — shrinking them toward zero  
to reduce overfitting and handle multicollinearity.

```
Linear Regression Loss:  L = Σ(yᵢ − ŷᵢ)²
Ridge Regression Loss:   L = Σ(yᵢ − ŷᵢ)² + λ × Σβⱼ²
                                              └─── L2 penalty
```

> 💡 "Ridge does not eliminate features — it **shrinks all coefficients** equally.  
>      When features are correlated, Ridge distributes the coefficient weight  
>      among them rather than arbitrarily assigning it to one."

---

## 🔍 When to Use Ridge Regression?

| Condition | Use Ridge? |
|-----------|:---------:|
| Linear relationship between X and y | ✅ Yes |
| Multicollinearity between features | ✅ Yes — Ridge's primary use case |
| Many features, some potentially irrelevant | ⚠️ Prefer Lasso for selection |
| Want to keep ALL features | ✅ Yes — Ridge never zeroes coefficients |
| Large number of features, all moderately useful | ✅ Yes |
| Need exact feature selection (zero out some) | ❌ No → Use Lasso |
| Target is categorical | ❌ No → Use Logistic Regression |

---

## 🧮 The Model

### Loss Function (Objective)

```
Ridge Loss = RSS + λ × L2 Penalty

         n                    p
L(β) =  Σ (yᵢ − ŷᵢ)²  +  λ × Σ βⱼ²
        i=1                  j=1

Where:
  RSS  = Residual Sum of Squares (same as Linear Regression)
  λ    = regularization strength (hyperparameter, called alpha in sklearn)
  βⱼ   = model coefficients (intercept β₀ is NOT penalized)
  p    = number of features
```

### The Tradeoff

```
λ = 0    →  Pure OLS (no regularization, same as Linear Regression)
λ → ∞   →  All βⱼ → 0 (model predicts the mean, maximum underfitting)
Optimal λ lies between these extremes

Small λ:  Low bias, high variance (close to OLS)
Large λ:  High bias, low variance (heavily shrunk coefficients)
```

---

## 🔧 Analytical Solution

Ridge has a **closed-form solution** (unlike Lasso):

```
β_ridge = (XᵀX + λI)⁻¹ Xᵀy

Where:
  XᵀX   = Gram matrix (n_features × n_features)
  λI     = Identity matrix scaled by λ
  Xᵀy    = cross-product of features and target

Key insight:
  XᵀX + λI is ALWAYS invertible even when XᵀX is singular
  This is why Ridge solves the multicollinearity problem!
```

**Comparison with OLS:**
```
OLS:   β = (XᵀX)⁻¹ Xᵀy          ← fails when XᵀX is singular
Ridge: β = (XᵀX + λI)⁻¹ Xᵀy     ← always works ✅
```

---

## 📐 Geometric Interpretation

Ridge regression constrains coefficients to lie **within an L2 ball** (circle in 2D):

```
            β₂
             │
             │    ●  OLS solution (unconstrained)
             │   /
             │  /  ← Ridge solution (where ellipse
             │ ○      touches the L2 constraint circle)
             │/
─────────────○──────────────── β₁
            /│
           / │
          /  │

L2 constraint: β₁² + β₂² ≤ t  (circle)
Loss contours: ellipses centered at OLS solution

Ridge solution = point where loss contour first touches the L2 circle
→ NEVER at an axis → coefficients NEVER exactly zero
```

**Compare with Lasso (L1 ball = diamond):**
```
Lasso diamond has corners on axes → solution CAN be exactly zero
Ridge circle has no corners → solution is NEVER exactly zero
```

---

## 🎛️ Alpha (λ) — The Regularization Parameter

```
sklearn parameter: alpha  (equivalent to λ in formulas)

alpha = 0.0    →  No regularization (= Linear Regression)
alpha = 0.01   →  Very mild regularization
alpha = 1.0    →  Moderate regularization (common default)
alpha = 10.0   →  Strong regularization
alpha = 1000.0 →  Very strong regularization (coefficients ≈ 0)
```

**How to choose alpha:**
```
Method 1: RidgeCV — built-in cross-validation over a range of alphas
Method 2: GridSearchCV — manual CV grid search
Method 3: Validation curve — plot val score vs alpha

ALWAYS search on a log scale:
  alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
```

---

## 📊 Effect on Coefficients

```
Feature   | OLS Coef | Ridge (α=0.1) | Ridge (α=1) | Ridge (α=100)
──────────────────────────────────────────────────────────────────
Feature_1 |   2.50   |     2.45      |    2.10     |     0.80
Feature_2 |   1.80   |     1.76      |    1.55     |     0.65
Feature_3 |   0.05   |     0.04      |    0.03     |     0.01
Feature_4 | -3.20    |    -3.10      |   -2.70     |    -1.10

Observations:
  - All coefficients shrink toward 0 as α increases
  - No coefficient reaches exactly 0 (unlike Lasso)
  - Larger original coefficients shrink more in absolute terms
  - Small coefficients shrink toward 0 fastest in relative terms
```

---

## 🆚 Ridge vs Linear Regression vs Lasso

| Aspect | Linear Regression | Ridge (L2) | Lasso (L1) |
|--------|:-----------------:|:----------:|:----------:|
| Penalty | None | Σβⱼ² | Σ\|βⱼ\| |
| Coefficients | Unregularized | Shrinks toward 0 | Shrinks to exactly 0 |
| Feature Selection | ❌ No | ❌ No | ✅ Yes |
| Multicollinearity | ❌ Fails | ✅ Handles well | ⚠️ Picks one arbitrarily |
| Solution | Analytical | Analytical | Coordinate descent |
| Interpretability | ✅ High | ✅ High | ✅ High + sparse |
| Best For | No issues | Correlated features | Many irrelevant features |

---

## 📈 Bias-Variance Tradeoff in Ridge

```
As α increases:

Bias²       ↑  (model becomes simpler, misses patterns)
Variance    ↓  (predictions become more stable)
Test Error  ↓  then ↑  (U-shaped curve)

Error
│         OLS
│ ●──────────────────────────── Variance
│      ●──────────────────────
│           ●───────────────── Total Error (U-shape)
│                ●──────────
│                     ●──────────────────────────── Bias²
└──────────────────────────────── α (lambda)
  0    0.1   1    10   100
```

The **optimal alpha minimizes total test error** (bias² + variance).

---

## 🔧 sklearn Implementation

```python
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Option 1: Ridge with fixed alpha
model = Ridge(alpha=1.0)

# Option 2: RidgeCV — built-in cross-validation
model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
model.fit(X_train, y_train)
print(f'Best alpha: {model.alpha_}')

# Option 3: Pipeline (recommended — prevents leakage)
pipe = Pipeline([
    ('scaler', StandardScaler()),    # MUST scale before Ridge
    ('model',  Ridge(alpha=1.0))
])
```

**Why must you scale features before Ridge?**
```
Ridge penalty:  λ × (β₁² + β₂² + ... + βₚ²)

If Feature_1 is in range [0, 1]    → β₁ is large (model scales up)
If Feature_2 is in range [0, 1000] → β₂ is small (model scales down)

Without scaling: Ridge penalizes large-range features MORE
With scaling   : Ridge penalizes all features EQUALLY ✅
```

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Not scaling before Ridge | Unequal penalization | Always StandardScale first |
| Using Ridge when Lasso is needed | No feature selection | Check if sparsity matters |
| Searching alpha on linear scale | Misses important range | Search on log scale |
| Penalizing the intercept | Biased predictions | sklearn excludes intercept by default ✅ |
| Not using Cross-Validation for alpha | Overfits to validation | Use RidgeCV or GridSearchCV |
| Assuming Ridge = feature selection | Coefficients never zero | Use Lasso for selection |

---

## 🔗 Related Topics

- `Linear_Regression` — Foundation that Ridge extends with L2 penalty
- `Lasso_Regression` — L1 regularization with feature selection
- `ElasticNet` — Combined L1 + L2 regularization
- `Polynomial_Regression` — Apply Ridge after polynomial expansion
- `07_Hyperparameter_Tuning` — GridSearchCV for optimal alpha
- `06_Feature_Selection` — Use Ridge coefficient magnitudes for ranking

---

## 📚 References

- Scikit-learn `Ridge`: [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- Scikit-learn `RidgeCV`: [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html)
- The Elements of Statistical Learning — Chapter 3.4
- An Introduction to Statistical Learning — Chapter 6.2
