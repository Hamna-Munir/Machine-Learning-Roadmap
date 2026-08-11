# 📘 Time Series Preprocessing — Theory

---

## 📌 What is Time Series Preprocessing?

Time series preprocessing transforms raw temporal data into a format  
suitable for machine learning — handling the **ordered, dependent, and  
often non-stationary** nature of time-indexed observations.

```
Raw Time Series:                   After Preprocessing:
  t₁: 120                            Stationary, scaled, feature-rich
  t₂: 135   →  preprocessing  →      ready for ML / forecasting models
  t₃: 128
  ...
  tₙ: 142

Key challenges vs standard ML:
  ❌ Observations are NOT independent — past influences future
  ❌ Standard train/test split leaks future into past
  ❌ Many series are non-stationary (trend, seasonality)
  ❌ Missing timestamps must be handled carefully
  ❌ Feature engineering must respect temporal order
```

---

## 🔍 Why Time Series Preprocessing is Different

```
Standard ML:               Time Series ML:
  Shuffle data ✅            Never shuffle — order matters ❌
  Random split ✅            Walk-forward / expanding split ❌
  i.i.d. samples ✅          Autocorrelated samples ❌
  Scale once ✅              Scale per fold (no future leakage) ❌
  Any features ✅            Only past features (no lookahead) ❌
```

---

## 🗂️ Core Preprocessing Steps

---

### 1. Parsing and Indexing

```python
import pandas as pd

df = pd.read_csv('data.csv', parse_dates=['date'], index_col='date')
df = df.sort_index()                      # ensure chronological order
df = df.asfreq('D')                       # set frequency (daily)
df.index.freq                             # confirm frequency

# Useful time components
df['year']    = df.index.year
df['month']   = df.index.month
df['day']     = df.index.day
df['weekday'] = df.index.dayofweek       # 0=Mon, 6=Sun
df['quarter'] = df.index.quarter
df['week']    = df.index.isocalendar().week
```

---

### 2. Handling Missing Values

```
Missing values in time series require special treatment:
  → Simple mean/median fill ignores temporal structure
  → Forward fill (ffill) uses last known value
  → Backward fill (bfill) uses next known value
  → Interpolation uses surrounding values

Methods:
  df.ffill()                    # forward fill (most common for finance)
  df.bfill()                    # backward fill
  df.interpolate(method='linear')     # linear interpolation
  df.interpolate(method='time')       # time-aware interpolation
  df.interpolate(method='spline', order=3)  # smooth curve

Rules:
  Short gaps (1–3 points)   → linear interpolation
  Long gaps (> 7 points)    → investigate — may be structural break
  Leading/trailing NaN      → drop or handle separately
  Seasonal gaps             → seasonal decomposition first
```

---

### 3. Stationarity — The Most Critical Concept

```
A time series is STATIONARY if its statistical properties
(mean, variance, autocorrelation) do NOT change over time.

Most ML models assume stationarity — non-stationary data causes
spurious relationships and unreliable forecasts.

Types of non-stationarity:
  Trend         → mean changes over time (upward/downward)
  Seasonality   → periodic patterns (daily, weekly, yearly)
  Heteroskedasticity → variance changes over time
  Structural break  → sudden change in level/trend

Visual checks:
  Plot the series → look for trend, changing variance
  Rolling statistics → rolling mean and std should be flat

Statistical tests:
  ADF (Augmented Dickey-Fuller):
    H₀: series has unit root (non-stationary)
    p < 0.05 → reject H₀ → stationary ✅

  KPSS (Kwiatkowski-Phillips-Schmidt-Shin):
    H₀: series is stationary
    p < 0.05 → reject H₀ → non-stationary ❌

  Use both: ADF stationary + KPSS stationary → confirmed stationary
```

---

### 4. Making a Series Stationary

#### Differencing
```python
# First-order differencing (removes linear trend)
df['diff1'] = df['value'].diff(1)

# Second-order differencing (removes quadratic trend)
df['diff2'] = df['value'].diff(1).diff(1)

# Seasonal differencing (removes seasonality of period s)
df['diff_seasonal'] = df['value'].diff(12)  # monthly data, yearly seasonality

# Combined: seasonal + first-order
df['diff_both'] = df['value'].diff(12).diff(1)

# Undo differencing (inverse transform for predictions)
df['undiff'] = df['diff1'].cumsum() + df['value'].iloc[0]
```

#### Log Transformation
```python
import numpy as np

# Stabilizes variance (good for exponential growth)
df['log_value'] = np.log(df['value'])           # requires value > 0
df['log1p_value'] = np.log1p(df['value'])       # handles zero values

# Undo log transform
df['original'] = np.exp(df['log_value'])
```

#### Box-Cox Transformation
```python
from scipy.stats import boxcox

df['bc_value'], lambda_ = boxcox(df['value'])   # auto-selects optimal lambda
# lambda=0 → log transform; lambda=0.5 → sqrt transform
```

---

### 5. Decomposition — Trend, Seasonality, Residual

```
Classical decomposition splits a series into:
  yₜ = Tₜ + Sₜ + Rₜ   (additive)
  yₜ = Tₜ × Sₜ × Rₜ   (multiplicative — use when variance grows with level)

Where:
  Tₜ = Trend component
  Sₜ = Seasonal component
  Rₜ = Residual (remainder)

Use additive when:     seasonal amplitude is CONSTANT
Use multiplicative when: seasonal amplitude GROWS with the trend

```python
from statsmodels.tsa.seasonal import seasonal_decompose, STL

# Classical decomposition
result = seasonal_decompose(df['value'], model='additive', period=12)
trend    = result.trend
seasonal = result.seasonal
residual = result.resid

# STL decomposition (more robust — handles outliers)
stl = STL(df['value'], period=12, robust=True)
res = stl.fit()
trend    = res.trend
seasonal = res.seasonal
residual = res.resid
```

---

### 6. Feature Engineering for Time Series ML

```
Lag Features (most important):
  df['lag_1']  = df['value'].shift(1)   # value 1 period ago
  df['lag_7']  = df['value'].shift(7)   # value 7 periods ago
  df['lag_28'] = df['value'].shift(28)  # value 28 periods ago

  Rule: only use lags ≥ forecast horizon to avoid lookahead bias!
  Forecasting 1-step ahead → use lags 1, 2, 3, ...
  Forecasting 7-step ahead → use lags 7, 8, 9, ... (not 1–6!)

Rolling Window Statistics:
  df['roll_mean_7']  = df['value'].rolling(7).mean()
  df['roll_std_7']   = df['value'].rolling(7).std()
  df['roll_min_7']   = df['value'].rolling(7).min()
  df['roll_max_7']   = df['value'].rolling(7).max()
  df['roll_range_7'] = df['roll_max_7'] - df['roll_min_7']

Expanding Window (cumulative):
  df['exp_mean'] = df['value'].expanding().mean()
  df['exp_std']  = df['value'].expanding().std()

Calendar Features:
  df['is_weekend']  = (df.index.dayofweek >= 5).astype(int)
  df['is_month_end']= df.index.is_month_end.astype(int)
  df['month_sin']   = np.sin(2*np.pi*df.index.month/12)   # cyclical encoding
  df['month_cos']   = np.cos(2*np.pi*df.index.month/12)
  df['hour_sin']    = np.sin(2*np.pi*df.index.hour/24)

  ⚠️ Use sin/cos encoding for cyclical features (month, hour, weekday)
     to preserve the circular nature: December is close to January

Target Encoding (lag-based):
  df['pct_change_1'] = df['value'].pct_change(1)    # % change
  df['diff_7_14']    = df['value'].shift(7) - df['value'].shift(14)
```

---

### 7. Train/Test Split — No Leakage

```
❌ WRONG — Standard random split leaks future into past:
  X_train, X_test, y_train, y_test = train_test_split(X, y)  # NEVER do this!

✅ CORRECT — Temporal split (all train before all test):
  split_idx = int(len(df) * 0.8)
  train = df.iloc[:split_idx]
  test  = df.iloc[split_idx:]

✅ CORRECT — Walk-forward validation (for CV):
  from sklearn.model_selection import TimeSeriesSplit
  tscv = TimeSeriesSplit(n_splits=5, gap=0)
  for train_idx, test_idx in tscv.split(X):
      ...

TimeSeriesSplit with gap:
  tscv = TimeSeriesSplit(n_splits=5, gap=7)
  # gap=7 → 7 samples between train and test per fold
  # Prevents data leakage when forecast horizon > 1
```

---

### 8. Scaling — Fit on Train Only

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit ONLY on train
X_test_sc  = scaler.transform(X_test)        # transform test — NO refit!

# ⚠️ In walk-forward CV: refit scaler inside each fold
for train_idx, test_idx in tscv.split(X):
    scaler = StandardScaler()
    X_tr_fold = scaler.fit_transform(X[train_idx])
    X_te_fold = scaler.transform(X[test_idx])
```

---

### 9. Autocorrelation Analysis

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.stattools import durbin_watson

# ACF: correlation of series with its own lags
# → Identifies MA order (q) for ARIMA
plot_acf(df['value'], lags=40)

# PACF: partial correlation after removing shorter-lag effects
# → Identifies AR order (p) for ARIMA
plot_pacf(df['value'], lags=40)

# Durbin-Watson test for autocorrelation in residuals
# DW ≈ 2 → no autocorrelation
# DW < 2 → positive autocorrelation
# DW > 2 → negative autocorrelation
dw = durbin_watson(residuals)
```

---

### 10. Outlier Detection and Treatment

```python
# IQR-based detection (time-aware)
Q1  = df['value'].rolling(90).quantile(0.25)
Q3  = df['value'].rolling(90).quantile(0.75)
IQR = Q3 - Q1
outliers = (df['value'] < Q1 - 1.5*IQR) | (df['value'] > Q3 + 1.5*IQR)

# Z-score based (rolling)
roll_mean = df['value'].rolling(30).mean()
roll_std  = df['value'].rolling(30).std()
z_score   = (df['value'] - roll_mean) / roll_std
outliers  = z_score.abs() > 3

# Treatment options:
df.loc[outliers, 'value'] = np.nan
df['value'] = df['value'].interpolate(method='linear')  # then interpolate
```

---

## 📊 Complete Preprocessing Checklist

```
□ 1. Parse dates → set as DatetimeIndex → sort chronologically
□ 2. Set frequency (asfreq) → fill missing timestamps
□ 3. Handle missing values (ffill / interpolate)
□ 4. Visual EDA → plot series, rolling stats, seasonal patterns
□ 5. Stationarity tests (ADF + KPSS)
□ 6. Make stationary if needed (differencing, log, Box-Cox)
□ 7. Decompose (trend, seasonality, residual)
□ 8. Detect and treat outliers
□ 9. Engineer lag features (respect forecast horizon!)
□ 10. Engineer rolling window features (mean, std, min, max)
□ 11. Add calendar features with cyclical encoding
□ 12. Temporal train/test split (no shuffling!)
□ 13. Scale AFTER split — fit only on train
□ 14. Drop NaN rows created by lags/rolling windows
```

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Random train/test split | Future data leaks into training set | Use temporal split only |
| Using lag_1 for 7-step forecast | Lookahead bias | Use lags ≥ forecast horizon |
| Fitting scaler on full data | Data leakage | Fit scaler only on train fold |
| Ignoring stationarity | Spurious correlations, bad forecasts | Test and transform to stationarity |
| Not checking autocorrelation | Residuals still contain structure | Plot ACF/PACF of residuals |
| Dropping NaN rows before split | Alignment issues | Drop after feature engineering |
| Using future calendar features | Lookahead | Calendar features (date) are ok — values from future are not |

---

## 🔗 Related Topics

- `09_Time_Series_Machine_Learning/feature_engineering.md` — Lag and rolling features in depth
- `09_Time_Series_Machine_Learning/forecasting_models.ipynb` — ARIMA, Prophet, ML models
- `05_Model_Evaluation/cross_validation.ipynb` — TimeSeriesSplit for CV
- `06_Feature_Selection/` — Feature selection for time series features

---

## 📚 References

- Pandas Time Series: [https://pandas.pydata.org/docs/user_guide/timeseries.html](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- Statsmodels TSA: [https://www.statsmodels.org/stable/tsa.html](https://www.statsmodels.org/stable/tsa.html)
- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (free): [https://otexts.com/fpp3/](https://otexts.com/fpp3/)
- sklearn `TimeSeriesSplit`: [https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
