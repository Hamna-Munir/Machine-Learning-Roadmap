# 📘 Linear Regression — Theory

---

## 📌 What is Linear Regression?

Linear Regression is the **simplest and most foundational supervised learning algorithm** —  
it models the relationship between one or more input features and a **continuous numerical target**  
by fitting a straight line (or hyperplane) through the data.

```
Simple Linear Regression:     ŷ = β₀ + β₁x
Multiple Linear Regression:   ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

> 💡 "All models are wrong, but some are useful."  
>      Linear Regression is the benchmark every other model is compared against.

---

## 🔍 When to Use Linear Regression?

| Condition | Use Linear Regression? |
|-----------|:---------------------:|
| Target is continuous (price, temperature, salary) | ✅ Yes |
| Relationship between X and y is approximately linear | ✅ Yes |
| Need interpretable coefficients | ✅ Yes |
| Quick baseline model needed | ✅ Yes |
| Target is categorical (0/1) | ❌ No → Use Logistic Regression |
| Relationship is strongly non-linear | ❌ No → Use Polynomial or Tree models |
| Heavy outliers in data | ⚠️ Caution → Use Robust Regression |

---

## 🧮 The Model

### Simple Linear Regression (1 feature)

```
ŷ = β₀ + β₁x

Where:
  ŷ  = predicted value
  β₀ = intercept (value of ŷ when x = 0)
  β₁ = slope (change in ŷ for a unit change in x)
  x  = input feature
```

### Multiple Linear Regression (n features)

```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

Matrix form:
  ŷ = Xβ

Where:
  X = design matrix (n_samples × n_features + 1)
  β = coefficient vector
```

---

## 🎯 Objective — Ordinary Least Squares (OLS)

Linear Regression minimizes the **Residual Sum of Squares (RSS)**:

```
RSS = Σᵢ (yᵢ − ŷᵢ)²  =  Σᵢ (yᵢ − β₀ − β₁x₁ᵢ − ... − βₙxₙᵢ)²

Residual = actual − predicted = yᵢ − ŷᵢ
```

**Geometric Interpretation:**
```
  y
  │       ●  ← actual point
  │      /|
  │     / | ← residual (error)
  │    /  ●  ← predicted point on line
  │   /
  │  / ← regression line
  └──────── x

Goal: minimize the total squared vertical distance from all points to the line
```

---

## 🔧 Solving for Coefficients

### 1. Analytical Solution (Normal Equation)

```
β = (XᵀX)⁻¹Xᵀy

Advantages:
  - Exact solution in one step
  - No learning rate to tune

Disadvantages:
  - O(n³) time complexity for matrix inversion
  - Fails when XᵀX is singular (non-invertible)
  - Slow for very large datasets (n_features > 10,000)
```

### 2. Gradient Descent (Iterative Optimization)

Iteratively updates coefficients in the direction that reduces the loss:

```
Step 1: Initialize β randomly (or at zero)
Step 2: Compute gradient of loss w.r.t. β
Step 3: Update β:  βⱼ ← βⱼ − α × ∂Loss/∂βⱼ
Step 4: Repeat until convergence

Where α = learning rate (step size)

Gradient:  ∂MSE/∂βⱼ = −(2/n) × Σ xᵢⱼ(yᵢ − ŷᵢ)
```

**Gradient Descent Variants:**
```
Batch GD      → Use all n samples per update (stable, slow)
Stochastic GD → Use 1 sample per update (fast, noisy)
Mini-Batch GD → Use k samples per update (best balance)
```

---

## 📐 Key Assumptions

Linear Regression makes **4 core assumptions** — violating them affects coefficient reliability:

### 1. Linearity
The relationship between X and y must be **linear**.
```
Check: Scatter plot of X vs y, residual plot
Fix  : Apply polynomial features, log transform
```

### 2. Independence of Errors
Residuals must be **independent** of each other — no autocorrelation.
```
Check: Durbin-Watson test, ACF plot of residuals
Fix  : Add lag features, use time-series models
```

### 3. Homoscedasticity (Constant Variance)
The variance of residuals must be **constant** across all predicted values.
```
Check: Residuals vs Fitted plot — should show random scatter
Fix  : Log/sqrt transform of target, Weighted Least Squares

Homoscedastic:              Heteroscedastic:
residuals                   residuals
   ●  ● ●  ●  ●               ●           ●●●
 ●   ●   ●   ●              ●          ●●  ●●
   ●  ●  ●  ●  ●           ●        ●●●  ●
─────────────────          ─────────────────
     fitted ŷ                  fitted ŷ (fan shape)
```

### 4. Normality of Residuals
Residuals should be **approximately normally distributed** (for valid inference).
```
Check: Q-Q plot of residuals, Shapiro-Wilk test
Fix  : Transform target variable, remove outliers
```

### 5. No Multicollinearity
Features should **not be highly correlated** with each other.
```
Check: Correlation matrix, VIF (VIF > 10 = problem)
Fix  : Drop one of the correlated features, use Ridge/PCA
```

---

## 📊 Model Evaluation Metrics

### R² (Coefficient of Determination)
```
R² = 1 − (SS_res / SS_tot)

SS_res = Σ(yᵢ − ŷᵢ)²   ← Residual Sum of Squares
SS_tot = Σ(yᵢ − ȳ)²    ← Total Sum of Squares

R² = 0.0 → Model explains none of the variance
R² = 1.0 → Model explains all of the variance
R² < 0   → Model worse than predicting the mean ❌
```

### Adjusted R²
```
Adj. R² = 1 − [(1 − R²)(n − 1) / (n − k − 1)]

Where:
  n = number of samples
  k = number of features

Penalizes adding features that don't improve the model
Always use Adj. R² when comparing models with different numbers of features
```

### Mean Absolute Error (MAE)
```
MAE = (1/n) × Σ|yᵢ − ŷᵢ|

Units: same as target variable
Interpretation: average absolute prediction error
Robust to outliers (unlike MSE)
```

### Mean Squared Error (MSE) & RMSE
```
MSE  = (1/n) × Σ(yᵢ − ŷᵢ)²
RMSE = √MSE

MSE penalizes large errors more heavily than MAE
RMSE is in the same units as the target
Lower is better for both
```

---

## 📈 Interpreting Coefficients

```
ŷ = 10,000 + 2,500 × Experience + 500 × Score

β₀ = 10,000  → Baseline salary with 0 experience and 0 score
β₁ = 2,500   → Each additional year of experience adds £2,500 to salary
β₂ = 500     → Each additional score point adds £500 to salary
```

**Important:** Coefficients are only directly comparable when features are **standardized**.  
Raw coefficient magnitude does NOT indicate feature importance.

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Not checking linearity | Model fits poorly | Plot X vs y before fitting |
| Not standardizing for comparison | Coefficients not comparable | Standardize features |
| Using R² alone | R² always increases with more features | Use Adjusted R² |
| Ignoring outliers | Coefficients pulled toward extremes | Detect with Cook's Distance |
| Multicollinearity | Unstable, unreliable coefficients | Check VIF, use Ridge |
| Extrapolating beyond training range | Predictions become unreliable | Never extrapolate |
| Not checking residuals | Hidden assumption violations | Always plot residuals |

---

## 🆚 Linear Regression vs Other Models

| Aspect | Linear Regression | Ridge | Lasso | Tree Models |
|--------|:----------------:|:-----:|:-----:|:-----------:|
| Regularization | ❌ | L2 | L1 | ❌ |
| Feature Selection | ❌ | ❌ | ✅ | ✅ |
| Handles Nonlinearity | ❌ | ❌ | ❌ | ✅ |
| Interpretability | ✅ High | ✅ High | ✅ High | ⚠️ Medium |
| Sensitive to Outliers | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No |

---

## 🔗 Related Topics

- `Polynomial_Regression` — Add non-linear terms to Linear Regression
- `Ridge_Regression` — L2 regularization to handle multicollinearity
- `Lasso_Regression` — L1 regularization for feature selection
- `ElasticNet` — Combined L1 + L2 regularization
- `05_Model_Evaluation/regression_metrics` — Full evaluation guide
- `06_Feature_Selection` — Select features before fitting

---

## 📚 References

- Scikit-learn `LinearRegression`: [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- Statsmodels OLS: [https://www.statsmodels.org/stable/regression.html](https://www.statsmodels.org/stable/regression.html)
- The Elements of Statistical Learning (Hastie, Tibshirani, Friedman) — Chapter 3
- An Introduction to Statistical Learning (James et al.) — Chapter 3
