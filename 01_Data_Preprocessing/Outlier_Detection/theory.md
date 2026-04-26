# 📘 Outlier Detection — Theory

---

## 📌 What is an Outlier?

An **outlier** is a data point that deviates significantly from the rest of the dataset — it is unusually far from the expected range or pattern.

```
Normal Data:   10, 12, 11, 13, 12, 14, 11, 13
With Outlier:  10, 12, 11, 13, 12, 14, 11, 13, 987  ← outlier
```

---

## 🔍 Why Do Outliers Matter?

| Effect | Impact |
|--------|--------|
| **Skews statistics** | Mean and std shift drastically |
| **Distorts models** | Linear Regression coefficients pulled toward outlier |
| **Breaks assumptions** | Normality, homoscedasticity violated |
| **Hurts scaling** | Min-Max scaler compresses all values |
| **Misleads clusters** | K-Means centroids pulled off center |

> ✅ However, outliers can also be **signals** — fraud detection, rare disease diagnosis, network intrusion. Always understand context before removing.

---

## 🏷️ Types of Outliers

### 1. Point Outliers (Global)
A single observation far from the rest of the data.
```
Data: [10, 11, 12, 13, 500]   ← 500 is a global outlier
```

### 2. Contextual Outliers (Conditional)
Normal in one context, abnormal in another.
```
Temperature 35°C → Normal in summer, Outlier in winter
```

### 3. Collective Outliers
A group of observations that are outliers together, but individually seem normal.
```
Individual transaction of $200 → Normal
100 transactions of $200 in 1 minute → Collective outlier (fraud)
```

---

## 🛠️ Detection Techniques

---

### 1️⃣ Z-Score Method

Measures how many **standard deviations** a point is from the mean.

**Formula:**
```
Z = (X - μ) / σ

Where:
  μ = mean
  σ = standard deviation
```

**Rule:** Points with |Z| > 3 are considered outliers (covers 99.7% of normal distribution).

**Assumptions:**
- Data is approximately **normally distributed**
- Sensitive to extreme values (mean and std are non-robust)

**When to use:**
- Normally distributed, univariate data
- Quick initial screening

---

### 2️⃣ IQR Method (Interquartile Range)

Uses the **middle 50%** of data to define the normal range.

**Formula:**
```
IQR = Q3 - Q1

Lower Fence = Q1 - 1.5 × IQR
Upper Fence = Q3 + 1.5 × IQR

Points outside [Lower Fence, Upper Fence] → Outliers
```

**Visual:**
```
      |----[=====|=====]-------|
   Lower   Q1  Med  Q3    Upper
   Fence              Fence
    ●                              ●
(outlier)                      (outlier)
```

**Advantages:**
- **Robust** — not influenced by the outliers themselves
- Works on **non-normal** distributions
- The default method for **box plots**

**When to use:**
- Skewed distributions
- Exploratory data analysis
- Univariate outlier detection

---

### 3️⃣ Modified Z-Score (Median Absolute Deviation — MAD)

A more **robust** version of Z-Score using **median** instead of mean.

**Formula:**
```
MAD = median(|X - median(X)|)

Modified Z = 0.6745 × (X - median(X)) / MAD

Threshold: |Modified Z| > 3.5  → Outlier
```

**When to use:**
- Small datasets
- When Z-Score is too sensitive to its own outliers

---

### 4️⃣ Isolation Forest

An **ensemble tree-based** algorithm that isolates anomalies.

**Core Idea:**
- Anomalies are **easier to isolate** — they require fewer splits in a random tree
- Normal points need more splits to be isolated

```
Anomaly:               Normal Point:
    root                   root
     |                    /    \
   leaf (2 splits)      ...     ...
                        leaf (many splits)
```

**Parameters:**
- `contamination` : Expected proportion of outliers (e.g., 0.05 = 5%)
- `n_estimators`  : Number of trees (default: 100)

**Advantages:**
- Works in **high dimensions**
- **No distance/density** calculation needed
- Fast and scalable

**When to use:**
- High-dimensional data
- No assumptions about data distribution
- Large datasets

---

### 5️⃣ Local Outlier Factor (LOF)

A **density-based** method that compares local density of a point to its neighbors.

**Core Idea:**
- If a point's neighborhood density is **much lower** than its neighbors → outlier
- LOF score > 1 → outlier (the higher, the more anomalous)

```
Dense Region:    ● ● ● ●       LOF ≈ 1.0 (normal)
Sparse Point:            ●     LOF >> 1.0 (outlier)
```

**Parameters:**
- `n_neighbors`  : Number of neighbors to consider (default: 20)
- `contamination`: Expected proportion of outliers

**When to use:**
- Datasets with **varying density clusters**
- When global thresholds don't capture local anomalies

---

### 6️⃣ DBSCAN (Density-Based Spatial Clustering)

Clusters dense regions and labels low-density points as **noise (outliers)**.

**Core Idea:**
- **Core points**: Have at least `min_samples` neighbors within `eps` radius
- **Border points**: Within `eps` of a core point but fewer neighbors
- **Noise points**: Neither core nor border → **outliers (label = -1)**

**Parameters:**
- `eps`         : Radius of neighborhood
- `min_samples` : Minimum points to form a dense region

**When to use:**
- Spatial data
- When cluster shape is arbitrary (not spherical)

---

### 7️⃣ Elliptic Envelope

Fits a **Gaussian distribution** to the data and flags points outside the ellipse as outliers.

**Core Idea:**
- Assumes data comes from a **Gaussian distribution**
- Uses **Mahalanobis distance** to measure how far a point is from the distribution center

**When to use:**
- Normally distributed, multivariate data
- Low-dimensional feature space

---

### 8️⃣ Winsorization (Capping)

Does **not remove** outliers — **caps** them at a percentile boundary.

**Formula:**
```
Lower cap = percentile(X, p)        e.g., 5th percentile
Upper cap = percentile(X, 100 - p)  e.g., 95th percentile

Values below lower cap → replaced with lower cap
Values above upper cap → replaced with upper cap
```

**When to use:**
- When you want to **retain rows** but reduce outlier influence
- Regression models where row count matters

---

## 📊 Technique Comparison Table

| Technique | Type | Dimensionality | Distribution Assumption | Action |
|-----------|------|:--------------:|:-----------------------:|:------:|
| Z-Score | Statistical | Univariate | Normal ✅ | Remove |
| IQR Method | Statistical | Univariate | None ✅ | Remove |
| Modified Z-Score (MAD) | Statistical | Univariate | Robust ✅ | Remove |
| Isolation Forest | ML (Tree) | Multivariate | None ✅ | Remove |
| Local Outlier Factor | ML (Density) | Multivariate | None ✅ | Remove |
| DBSCAN | ML (Clustering) | Multivariate | None ✅ | Label |
| Elliptic Envelope | Statistical | Multivariate | Gaussian ⚠️ | Remove |
| Winsorization | Statistical | Univariate | None ✅ | Cap |

---

## 🧠 Decision Guide: Which Method to Use?

```
Univariate feature?
  ├── YES → Is data normally distributed?
  │           ├── YES → Z-Score or Modified Z-Score (MAD)
  │           └── NO  → IQR Method or Winsorization
  │
  └── NO (Multivariate) → What do you know about your data?
                            ├── Gaussian assumed   → Elliptic Envelope
                            ├── Varying densities  → Local Outlier Factor
                            ├── High-dimensional   → Isolation Forest
                            └── Spatial / Clusters → DBSCAN
```

---

## ⚠️ What to Do After Detecting Outliers?

| Action | When to Use |
|--------|-------------|
| **Remove** | Caused by data entry error or instrument failure |
| **Cap (Winsorize)** | Real but extreme — preserve row count |
| **Transform** | Log/Power transform to reduce influence |
| **Keep** | Outliers are the signal (fraud, anomaly detection) |
| **Separate model** | Train one model on normal data, one for outliers |

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Removing outliers blindly | May remove valid signal | Understand context first |
| Applying detection after split | Leaks test info | Detect on train, apply to test |
| Using Z-Score on skewed data | Mean/std are non-robust | Use IQR or MAD instead |
| Wrong `contamination` param | Over/under-detection | Tune based on domain knowledge |
| Treating all outliers as errors | May be real anomalies | Explore before removing |

---

## 🔗 Related Topics

- `Handling_Missing_Values` — Outliers can cause imputation bias
- `Feature_Scaling` — Remove/cap outliers **before** Min-Max scaling
- `EDA / Univariate_Analysis` — Box plots and histograms reveal outliers visually
- `04_Unsupervised_Learning / DBSCAN` — Full DBSCAN clustering coverage

---

## 📚 References

- Scikit-learn Outlier Detection: [https://scikit-learn.org/stable/modules/outlier_detection.html](https://scikit-learn.org/stable/modules/outlier_detection.html)
- `IsolationForest`: [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- `LocalOutlierFactor`: [https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)
- SciPy `zscore`: [https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zscore.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zscore.html)
