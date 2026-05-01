# 📘 Univariate Analysis — Theory

---

## 📌 What is Univariate Analysis?

Univariate Analysis is the **simplest form of data analysis** where we examine **one variable at a time**  
to understand its distribution, central tendency, spread, and shape.

```
Dataset has 10 features → Univariate Analysis = study each of the 10 features independently
```

> 💡 "Before asking how features relate to each other,  
>      first understand each feature on its own."

---

## 🔍 Why Univariate Analysis?

| Goal | What It Reveals |
|------|----------------|
| **Understand distribution** | Is the data normal, skewed, bimodal? |
| **Detect outliers** | Extreme values that may distort models |
| **Spot missing values** | Gaps in data that need imputation |
| **Assess data quality** | Errors, impossible values, inconsistencies |
| **Guide preprocessing** | Which scaling or transformation to apply |
| **Understand cardinality** | How many unique values does a feature have? |

---

## 🏷️ Types of Variables

```
Variables
    │
    ├── Numerical (Quantitative)
    │       ├── Continuous   → Height, Weight, Temperature, Price
    │       └── Discrete     → Age (in years), Count, Number of rooms
    │
    └── Categorical (Qualitative)
            ├── Nominal      → Color, City, Gender (no order)
            └── Ordinal      → Education level, Rating (ordered)
```

---

## 🛠️ Techniques for Numerical Variables

---

### 1️⃣ Measures of Central Tendency

Describe the **center** of the distribution.

| Measure | Formula | Best When |
|---------|---------|-----------|
| **Mean** | Σx / n | Symmetric distribution, no outliers |
| **Median** | Middle value when sorted | Skewed data, outliers present |
| **Mode** | Most frequently occurring value | Categorical or discrete data |

```
Example: Salaries = [30K, 35K, 40K, 38K, 500K]

Mean   = 128.6K  ← pulled heavily by the outlier (500K)
Median = 38K     ← robust to the outlier ✅
Mode   = No mode (all values are unique)
```

---

### 2️⃣ Measures of Spread (Dispersion)

Describe how **spread out** the values are around the center.

| Measure | Formula | Notes |
|---------|---------|-------|
| **Range** | Max − Min | Highly sensitive to outliers |
| **Variance** | Σ(x − μ)² / n | Units are squared |
| **Std Deviation** | √Variance | Same units as original data |
| **IQR** | Q3 − Q1 | Robust — not affected by outliers |
| **CV (Coeff. of Variation)** | (Std / Mean) × 100 | Relative spread as a percentage |

---

### 3️⃣ Shape of Distribution

| Measure | Description |
|---------|-------------|
| **Skewness** | Asymmetry of the distribution |
| **Kurtosis** | Tail heaviness (peakedness vs flatness) |

**Skewness:**
```
Negative Skew (Left):   long tail on the LEFT   → mean < median
Symmetric (Normal):     tails are balanced       → mean ≈ median
Positive Skew (Right):  long tail on the RIGHT   → mean > median

      Left Skew          Normal          Right Skew
         ████              ██               ██
       ██████            ██████           ██████
      ████████          ████████         ████████
     ██████████        ██████████       ██████████
   ─────────────      ─────────────    ─────────────
```

**Kurtosis:**
```
Platykurtic  (< 3): Flat distribution,  light tails  → fewer outliers
Mesokurtic   (= 3): Normal distribution (benchmark)
Leptokurtic  (> 3): Sharp peak,         heavy tails  → more outliers
```

---

### 4️⃣ Percentiles & Quartiles

Divide data into **equal-sized groups** to understand its spread.

```
               Q1        Q2        Q3
              (25%)    (50%)     (75%)
──────────────|─────────|──────────|────────────────
0%         25th pct   Median    75th pct        100%

IQR = Q3 − Q1   (covers the middle 50% of data)
```

| Percentile | Description |
|-----------|-------------|
| 25th (Q1) | 25% of data falls below this value |
| 50th (Q2) | Median — half below, half above |
| 75th (Q3) | 75% of data falls below this value |
| 90th, 95th, 99th | Used to detect extreme outliers |

---

## 📊 Visualization Techniques for Numerical Variables

---

### 📈 Histogram

Shows the **frequency distribution** of a continuous variable by grouping values into bins.

```
Frequency
    │   ██
    │  ████
    │ ██████
    │████████
    └──────────── Value bins
```

**Key decisions:**
- `bins` count affects shape — too few oversimplifies, too many adds noise
- **Rule of thumb:** `bins ≈ √n` or use `'auto'` in matplotlib

**What to look for:**
- Bell shape → normally distributed
- Right or left tail → skewed data
- Two peaks → bimodal (two distinct subgroups exist)
- Completely flat → uniform distribution

---

### 📦 Box Plot (Box-and-Whisker)

Summarizes distribution using the **5-number summary**: Min, Q1, Median, Q3, Max.

```
        ┌──────────────┐
        │              │
────────┤     IQR      ├────────    ← whiskers extend 1.5 × IQR
        │              │
        └──────────────┘
   Min  Q1    Median   Q3  Max       ●  ← outlier (beyond whisker)
```

**What to look for:**
- Length of box → spread of middle 50%
- Position of median line → symmetry or skew
- Whisker length → tail behavior
- Individual dots beyond whiskers → outliers

---

### 〰️ KDE Plot (Kernel Density Estimation)

A **smooth continuous curve** that estimates the probability density function of a variable.

```
Density
    │      ╭──╮
    │    ╭─╯  ╰─╮
    │  ╭─╯      ╰─╮
    │──╯            ╰──── Value
```

**Advantages over histogram:**
- Not sensitive to bin width choice
- Provides a smoother representation
- Easy to compare multiple distributions on one plot

---

### 🎻 Violin Plot

Combines a **KDE plot + box plot** — shows both the full distribution shape and key summary statistics.

```
         ╭───╮
        ╭─────╮
        │  ●  │  ← median
        ╰─────╯
         ╰───╯
```

**When to use:**
- Comparing distributions across multiple groups
- When you need both shape detail AND summary statistics

---

### 📐 Q-Q Plot (Quantile-Quantile)

Compares sample quantiles against a **theoretical normal distribution** to test normality.

```
Theoretical Quantiles
    │            ●●●
    │         ●●●
    │       ●●
    │    ●●●
    │●●●
    └──────────────────── Sample Quantiles

Points follow the diagonal  → normally distributed ✅
Points curve away           → non-normal distribution ❌
```

---

### 📊 ECDF (Empirical Cumulative Distribution Function)

Shows the **proportion of data** at or below each value — a step-wise cumulative view.

```
Cumulative %
1.0 ─────────────────────────────╮
                              ╭──╯
0.5 ─────────────────────╮ ╭──╯
                         ╰─╯
0.0 ──────────────────────────── Value
```

---

## 🛠️ Techniques for Categorical Variables

---

### 1️⃣ Frequency Table

Counts the **number and proportion** of occurrences for each category.

```
City       Count    Percentage
London       320       40%
Paris        240       30%
Berlin       160       20%
Tokyo         80       10%
```

---

### 2️⃣ Bar Chart

Displays the **frequency or proportion** of each unique category value.

```
Count
  │   ████
  │   ████   ████
  │   ████   ████   ████
  │   ████   ████   ████   ████
  └──────────────────────────── Categories
       A       B      C      D
```

**Horizontal bar chart:** preferred when category names are long.

---

### 3️⃣ Pie / Donut Chart

Shows the **proportional composition** of all categories.

**⚠️ Use sparingly:**
- Difficult to compare slices of similar size
- Best with 5 or fewer categories
- Bar charts are almost always clearer and more precise

---

### 4️⃣ Count Plot

A seaborn-specific bar chart for **counting categorical occurrences** — automatically handles grouping by hue.

---

## 📊 Summary: Which Statistics Apply?

| Statistic | Numerical | Categorical |
|-----------|:---------:|:-----------:|
| Count | ✅ | ✅ |
| Mean | ✅ | ❌ |
| Median | ✅ | ❌ |
| Mode | ✅ | ✅ |
| Std Dev | ✅ | ❌ |
| IQR | ✅ | ❌ |
| Skewness | ✅ | ❌ |
| Kurtosis | ✅ | ❌ |
| Frequency / % | ❌ | ✅ |
| Unique Count | ✅ | ✅ |

---

## 🧠 Decision Guide: Which Visualization to Use?

```
Variable Type?
    │
    ├── NUMERICAL
    │       ├── Understand distribution shape  → Histogram + KDE
    │       ├── Detect outliers quickly        → Box Plot
    │       ├── Check normality assumption     → Q-Q Plot
    │       ├── Compare group distributions    → Violin Plot / Overlapping KDE
    │       ├── Cumulative view                → ECDF
    │       └── Get all summary statistics     → df.describe()
    │
    └── CATEGORICAL
            ├── Count occurrences per class    → Bar Chart / Count Plot
            ├── Show proportional breakdown    → Pie / Donut Chart
            └── Inspect unique values          → value_counts()
```

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Using mean with outliers | Mean is pulled toward extremes | Report median alongside mean |
| Too few histogram bins | Hides true distribution shape | Use `bins=int(√n)` or `'auto'` |
| Ignoring skewness | Assumes normality incorrectly | Always check skewness before modeling |
| Pie charts with many slices | Visually confusing and hard to compare | Use a bar chart instead |
| Ignoring cardinality | High-cardinality treated as low | Always run `nunique()` first |
| Skipping missing value check | Misleading statistics | Start every EDA with `isnull().sum()` |

---

## 🔗 Related Topics

- `Handling_Missing_Values` — Univariate stats guide the right imputation strategy
- `Outlier_Detection` — Box plots and Z-score are rooted in univariate analysis
- `Feature_Scaling` — Skewness and distribution shape inform the scaler choice
- `Bivariate_Analysis` — Next step after understanding each feature individually
- `Feature_Engineering` — Univariate insights drive binning and transformation decisions

---

## 📚 References

- Pandas `describe()`: [https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html)
- Seaborn Distribution Plots: [https://seaborn.pydata.org/tutorial/distributions.html](https://seaborn.pydata.org/tutorial/distributions.html)
- Matplotlib Histogram: [https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html)
- SciPy Stats (skewness, kurtosis): [https://docs.scipy.org/doc/scipy/reference/stats.html](https://docs.scipy.org/doc/scipy/reference/stats.html)
- Statsmodels Q-Q Plot: [https://www.statsmodels.org/stable/generated/statsmodels.graphics.gofplots.qqplot.html](https://www.statsmodels.org/stable/generated/statsmodels.graphics.gofplots.qqplot.html)
