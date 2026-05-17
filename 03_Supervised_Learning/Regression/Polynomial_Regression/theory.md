# 📘 Polynomial Regression — Theory

---

## 📌 What is Polynomial Regression?

Polynomial Regression is an extension of Linear Regression that models **non-linear relationships**  
between features and the target by adding **polynomial terms** (squared, cubed, cross-products)  
to the feature space — while remaining a **linear model in its parameters**.

```
Linear Regression:      ŷ = β₀ + β₁x
Degree-2 Polynomial:    ŷ = β₀ + β₁x + β₂x²
Degree-3 Polynomial:    ŷ = β₀ + β₁x + β₂x² + β₃x³
```

> 💡 "Polynomial Regression fits a curve, not a straight line — but it is still  
>      a linear model because the coefficients β are linear, not the features."

---

## 🔍 When to Use Polynomial Regression?

| Condition | Use Polynomial Regression? |
|-----------|:-------------------------:|
| Relationship is curved (U-shape, S-shape) | ✅ Yes |
| Linear Regression underfits (high bias) | ✅ Yes |
| Moderate number of features | ✅ Yes |
| Quick non-linear baseline needed | ✅ Yes |
| Very high-dimensional data | ❌ No → Feature explosion |
| Extreme non-linearity | ❌ No → Use tree models |
| Many features + non-linearity | ❌ No → Use Random Forest / XGBoost |

---

## 🧮 The Model

### Single Feature — Degree d

```
ŷ = β₀ + β₁x + β₂x² + β₃x³ + ... + βdxᵈ

Example (degree = 2):
ŷ = β₀ + β₁x + β₂x²

This is still a LINEAR model because:
  - Linear in the PARAMETERS β₀, β₁, β₂
  - Non-linear only in the INPUT feature x
```

### Multiple Features — Degree 2

With 2 features [x₁, x₂], degree=2 expands to:

```
[1, x₁, x₂, x₁², x₁·x₂, x₂²]

  ↑       ↑           ↑
Bias  Original    Polynomial
      features    & interaction
                    terms
```

**General formula for number of output features:**
```
C(n + d, d) = (n + d)! / (n! × d!)

Where:
  n = number of input features
  d = polynomial degree

Examples:
  n=2, d=2 → 6 features
  n=5, d=2 → 21 features
  n=10, d=2 → 66 features   ← grows fast!
  n=10, d=3 → 286 features  ← explosion!
```

---

## 🎯 Objective — Same as Linear Regression (OLS)

```
Minimize RSS = Σᵢ (yᵢ − ŷᵢ)²

The polynomial features are created FIRST using PolynomialFeatures,
then standard OLS (or Gradient Descent) is applied to the expanded feature matrix.
```

**Two-step sklearn pipeline:**
```
Step 1:  PolynomialFeatures(degree=d)  →  Expand X to X_poly
Step 2:  LinearRegression()            →  Fit OLS on X_poly
```

---

## 🔧 Choosing the Degree

The degree `d` is the **key hyperparameter** — it controls the model's flexibility.

```
d = 1  →  Linear (straight line) — high bias, low variance
d = 2  →  Quadratic (parabola)   — balanced
d = 3  →  Cubic                  — more flexible
d ≥ 5  →  Danger zone: overfitting

Too LOW degree:                Too HIGH degree:
  Underfitting (high bias)       Overfitting (high variance)

  y                              y
  │     ●                        │ ●
  │   ●   ●   ●                  │   ●
  │ ─────────────                │ ─/\/\/\────
  │           ●  ●               │           ●●
  └────────────── x              └────────────── x
```

**How to choose degree:**
- Plot learning curves (train vs validation error vs degree)
- Use cross-validation to find degree with lowest validation error
- Apply regularization (Ridge) to control overfitting at higher degrees

---

## 📐 Bias-Variance Tradeoff

Polynomial Regression perfectly illustrates the **bias-variance tradeoff**:

```
         Bias²    Variance    Total Error
d = 1  │ High    │ Low      │ Underfits
d = 2  │ Medium  │ Medium   │ Good balance
d = 3  │ Low     │ Medium   │ Usually good
d = 5  │ Very Low│ High     │ Starts overfitting
d = 10 │ ~0      │ Very High│ Massively overfits

Optimal d = minimum point on the Total Error (U-shaped) curve
```

---

## 🔧 Interaction Features

`PolynomialFeatures` also creates **interaction terms** — products of different features.

```
Features: [Age, Experience]

Degree=2 expansion:
  [1, Age, Experience, Age², Age×Experience, Experience²]

Age × Experience = interaction term
→ "Does the effect of experience on salary depend on age?"
```

**To get ONLY interaction terms (no powers):**
```python
PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
# [Age, Experience] → [Age, Experience, Age×Experience]
```

---

## 📊 Model Evaluation Metrics

Same as Linear Regression:

| Metric | Formula | Notes |
|--------|---------|-------|
| **R²** | 1 − SS_res/SS_tot | Increases with degree — use Adj. R² |
| **Adj. R²** | Penalizes extra features | Fairer comparison across degrees |
| **MAE** | mean\|y − ŷ\| | Robust to outliers |
| **RMSE** | √MSE | Same units as target |
| **AIC/BIC** | Information criteria | Penalizes model complexity |

**⚠️ Important:**  
R² always increases as degree increases — it will reach ~1.0 on training data.  
Always use **test set / cross-validation R²** to detect overfitting.

---

## 📈 Learning Curves

Learning curves plot **train and validation error vs degree** (or training set size):

```
Error
  │
  │──── Train Error (decreases with degree)
  │           ╲
  │            ╲──────── Validation Error
  │                 ╲     (U-shape — first decreases, then increases)
  │                  ╲___/
  │
  └──────────────────────────── Degree
         1    2    3    4    5

Optimal degree = minimum validation error point
```

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| High degree without regularization | Severe overfitting | Use Ridge + PolynomialFeatures |
| Not scaling before polynomial expansion | Numerical instability (x¹⁰ = huge) | Always StandardScale after expansion |
| Using R² to choose degree | R² always increases | Use cross-validated R² or Adj. R² |
| Too many input features + high degree | Combinatorial explosion | Limit degree ≤ 3, use `interaction_only` |
| Extrapolating beyond training range | Polynomial curves explode at extremes | Never extrapolate polynomial models |
| Forgetting `include_bias=False` in pipeline | Duplicate intercept column | Set `include_bias=False` when using intercept |

---

## 🆚 Polynomial vs Other Non-Linear Models

| Aspect | Polynomial | Decision Tree | Random Forest | Neural Net |
|--------|:----------:|:-------------:|:-------------:|:----------:|
| Interpretability | ✅ High | ✅ High | ⚠️ Medium | ❌ Low |
| Handles many features | ❌ | ✅ | ✅ | ✅ |
| Extrapolation | ❌ Explodes | ❌ Flat | ❌ Flat | ⚠️ Varies |
| Training speed | ✅ Fast | ✅ Fast | ⚠️ Medium | ❌ Slow |
| Regularization | Via Ridge | Pruning | Bagging | Dropout/L2 |

---

## 🔗 Related Topics

- `Linear_Regression` — Foundation of Polynomial Regression
- `Ridge_Regression` — Regularize polynomial models to prevent overfitting
- `Lasso_Regression` — Feature selection on expanded polynomial feature set
- `Feature_Engineering` — Manual polynomial feature creation
- `07_Hyperparameter_Tuning` — Cross-validate degree selection

---

## 📚 References

- Scikit-learn `PolynomialFeatures`: [https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
- Scikit-learn Polynomial Regression Guide: [https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression](https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression)
- An Introduction to Statistical Learning (James et al.) — Chapter 7
- The Elements of Statistical Learning (Hastie et al.) — Chapter 5
