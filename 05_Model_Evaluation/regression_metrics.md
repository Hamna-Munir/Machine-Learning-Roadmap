# 📘 Regression Metrics — Theory

---

## 📌 What are Regression Metrics?

Regression metrics quantify how well a model's **continuous predictions**  
match the actual target values. Unlike classification, there is no simple  
"correct/incorrect" — instead, metrics measure the **magnitude and direction  
of prediction errors**.

```
Error = Actual − Predicted

Goal: minimize prediction errors across all samples

Key question: "How far off are my predictions, and does that matter?"
```

---

## 🔍 The Error Distribution

```
Perfect model:              Typical model:
  Residual                    Residual
      │                           │    ●
      │   ●●●●●●●●●              │  ●●●
      │   ●●●●●●●●●            ● │●●●●●●
      │   ●●●●●●●●●              │  ●●●●●
      │                         ●│   ●
      └──────────── ŷ            └──────────── ŷ

All residuals = 0              Residuals scattered around 0
(impossible in practice)       (ideal: symmetric, small variance)
```

---

## 📐 Core Regression Metrics

---

### 1. Mean Absolute Error (MAE)

```
MAE = (1/n) × Σ|yᵢ − ŷᵢ|

Where:
  yᵢ  = actual value
  ŷᵢ  = predicted value
  n   = number of samples

Properties:
  ✅ Same unit as target variable (interpretable)
  ✅ Robust to outliers (no squaring)
  ✅ Easy to explain: "predictions are off by X on average"
  ❌ Non-differentiable at 0 (gradient issues for optimization)
  ❌ Does not penalize large errors more than small ones

When to use:
  → Outliers present and you don't want them to dominate
  → Need a human-interpretable error in original units
  → Median-based prediction is more appropriate than mean

sklearn: mean_absolute_error(y_true, y_pred)
```

### 2. Mean Squared Error (MSE)

```
MSE = (1/n) × Σ(yᵢ − ŷᵢ)²

Properties:
  ✅ Differentiable — smooth gradient for optimization
  ✅ Penalizes large errors heavily (squared)
  ✅ Mathematically convenient
  ❌ Unit = target² (not directly interpretable)
  ❌ Sensitive to outliers (large errors magnified)
  ❌ Scale-dependent (comparing across datasets is misleading)

When to use:
  → You want to penalize large errors strongly
  → As a loss function during model training
  → Outliers are meaningful and should be penalized

sklearn: mean_squared_error(y_true, y_pred)
```

### 3. Root Mean Squared Error (RMSE)

```
RMSE = √MSE = √[(1/n) × Σ(yᵢ − ŷᵢ)²]

Properties:
  ✅ Same unit as target variable (interpretable)
  ✅ Still penalizes large errors heavily (via squaring before root)
  ✅ Most commonly reported metric in regression
  ❌ More sensitive to outliers than MAE
  ❌ Scale-dependent

Relationship to MAE:
  RMSE ≥ MAE always
  RMSE ≈ MAE → errors are uniformly distributed
  RMSE >> MAE → a few very large errors exist (outliers!)

When to use:
  → Default choice for continuous regression tasks
  → When large errors are more costly than small ones
  → When comparing models on the same dataset

sklearn: np.sqrt(mean_squared_error(y_true, y_pred))
         # or: mean_squared_error(y_true, y_pred, squared=False)
```

### 4. R² Score (Coefficient of Determination)

```
R² = 1 − SS_res / SS_tot

Where:
  SS_res = Σ(yᵢ − ŷᵢ)²    (residual sum of squares)
  SS_tot = Σ(yᵢ − ȳ)²     (total sum of squares)
  ȳ      = mean of y

Interpretation:
  R² = 1.0  → perfect fit (model explains all variance)
  R² = 0.0  → model is as good as predicting the mean (baseline)
  R² < 0.0  → model is worse than predicting the mean (very bad)
  R² = 0.80 → model explains 80% of variance in y

Properties:
  ✅ Scale-free — comparable across datasets
  ✅ Intuitive: "fraction of variance explained"
  ✅ Always between -∞ and 1.0
  ❌ Can be misleadingly high with many features
  ❌ Increases with more predictors (even irrelevant ones)
  ❌ Doesn't indicate whether the model is biased

When to use:
  → Comparing models on different datasets
  → Reporting overall model fit quality
  → Communication with non-technical stakeholders

sklearn: r2_score(y_true, y_pred)
```

### 5. Adjusted R²

```
Adj R² = 1 − [(1 − R²) × (n − 1)] / (n − p − 1)

Where:
  n = number of samples
  p = number of predictors (features)

Properties:
  ✅ Penalizes adding irrelevant features
  ✅ More reliable than R² for model comparison with different p
  ✅ Increases only if new feature genuinely improves fit
  ❌ Still scale-free like R²

Use: Adj R² over R² when comparing models with different numbers of features.
```

### 6. Mean Absolute Percentage Error (MAPE)

```
MAPE = (100/n) × Σ|yᵢ − ŷᵢ| / |yᵢ|

Properties:
  ✅ Scale-free (expressed as %)
  ✅ Easy to interpret: "predictions are X% off on average"
  ❌ Undefined when yᵢ = 0 (division by zero)
  ❌ Asymmetric: penalizes under-predictions more than over-predictions
  ❌ Poor for near-zero targets

When to use:
  → Comparing across different scales or units
  → Business contexts where % error is more meaningful
  → Time series forecasting

sklearn: mean_absolute_percentage_error(y_true, y_pred)
```

### 7. Median Absolute Error (MedAE)

```
MedAE = Median(|y₁ − ŷ₁|, |y₂ − ŷ₂|, ..., |yₙ − ŷₙ|)

Properties:
  ✅ Extremely robust to outliers
  ✅ Same unit as target (interpretable)
  ✅ Good for skewed error distributions
  ❌ Ignores magnitude of most errors
  ❌ Rarely used as primary metric

sklearn: median_absolute_error(y_true, y_pred)
```

### 8. Max Error

```
MaxError = max(|yᵢ − ŷᵢ|)

Properties:
  ✅ Captures worst-case prediction
  ✅ Critical for safety-sensitive applications
  ❌ Dominated by a single outlier
  ❌ Not a reliable average measure

sklearn: max_error(y_true, y_pred)
```

---

## 📊 Summary Comparison Table

| Metric | Unit | Outlier Sensitivity | Scale-Free | Use When |
|--------|:----:|:-------------------:|:----------:|----------|
| MAE | Same as y | Low | ❌ | Outliers present, need interpretability |
| MSE | y² | High | ❌ | Loss function, penalize large errors |
| RMSE | Same as y | High | ❌ | Default regression metric |
| R² | None | Medium | ✅ | Comparing across datasets |
| Adj R² | None | Medium | ✅ | Comparing models with different features |
| MAPE | % | Medium | ✅ | Business/forecasting contexts |
| MedAE | Same as y | Very Low | ❌ | Heavy outliers, skewed errors |
| Max Error | Same as y | Extreme | ❌ | Worst-case / safety-critical |

---

## 🔄 MAE vs RMSE — When Each Wins

```
MAE is preferred when:
  - Outliers exist and should NOT dominate the metric
  - All errors are equally important
  - End users need to understand "average error"

RMSE is preferred when:
  - Large errors are more costly than small ones
  - You want to penalize inconsistent predictions
  - Comparing with other models on the same scale

If RMSE >> MAE: outliers are present and dominating RMSE
If RMSE ≈ MAE: errors are roughly uniform in magnitude
```

---

## 🎯 Residual Analysis

Residuals = actual − predicted. A good model's residuals should be:

```
1. Centered at zero:     No systematic bias
2. Homoscedastic:        Constant variance across all ŷ values
3. Normally distributed: For confidence intervals to be valid
4. Independent:          No autocorrelation (especially in time series)

Residual Plots to make:
  a) Residuals vs Fitted:   Check homoscedasticity
  b) Q-Q Plot:              Check normality
  c) Scale-Location:        Check spread of residuals
  d) Residuals vs Leverage: Check influential points (Cook's distance)

⚠️ High residuals with high leverage → influential outliers → investigate!
```

---

## 📈 Choosing the Right Metric

```
Decision tree:

Is the target near zero or contains zeros?
  → Avoid MAPE
  → Use RMSE or MAE

Are large errors much worse than small ones?
  → RMSE (penalizes outliers)

Are errors roughly equally important?
  → MAE

Do you need scale-free comparison across datasets?
  → R² or MAPE

Are outliers present and should NOT dominate?
  → MAE or MedAE

Is interpretability to non-technical stakeholders key?
  → MAE (same units) or MAPE (percentage)
```

---

## 🔗 Related Topics

- `05_Model_Evaluation/classification_metrics.md` — Accuracy, F1, AUC for classifiers
- `05_Model_Evaluation/cross_validation.md` — CV strategies for reliable evaluation
- `03_Supervised_Learning/Regression/` — Linear, Ridge, Lasso models
- `07_Hyperparameter_Tuning/` — Use metrics as scoring in GridSearchCV

---

## 📚 References

- Scikit-learn Regression Metrics: [https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
- An Introduction to Statistical Learning — Chapter 3 (Linear Regression)
- The Elements of Statistical Learning — Chapter 2
