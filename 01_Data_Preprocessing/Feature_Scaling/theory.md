# 📘 Feature Scaling — Theory

---

## 📌 What is Feature Scaling?

Feature Scaling is the process of **normalizing or standardizing the range of independent variables** so that no single feature dominates the model due to its magnitude.

| Without Scaling | With Scaling |
|----------------|--------------|
| Age: 25–60 | Age: 0–1 |
| Salary: 20,000–200,000 | Salary: 0–1 |
| Model biased toward Salary ❌ | All features equally weighted ✅ |

---

## 🔍 Why Does Feature Scaling Matter?

- **Distance-based models** (KNN, SVM, K-Means) rely on Euclidean distance — larger-scale features dominate
- **Gradient descent** converges **faster** with scaled features — loss surface becomes symmetric
- **Regularization** (Ridge, Lasso) penalizes large coefficients — scaling ensures fair penalization
- **PCA / LDA** are variance-based — unscaled features skew principal components

### 📈 Visual Intuition: Gradient Descent

```
Without Scaling          With Scaling
(elongated ellipse)      (circular contour)

   ↑                          ↑
   │  ~~~                     │   ○○
   │ ~~~~~                    │  ○○○○
   │~~~~~~~                   │   ○○
   └──────→                   └──────→

Slow, zigzag path         Fast, direct path
```

---

## 🛠️ Scaling Techniques

---

### 1️⃣ Min-Max Scaling (Normalization)

Scales all values to a **fixed range**, typically [0, 1].

**Formula:**

```
X_scaled = (X - X_min) / (X_max - X_min)
```

**Custom range [a, b]:**

```
X_scaled = a + (X - X_min) * (b - a) / (X_max - X_min)
```

**Properties:**
- Output range: **[0, 1]** (or custom)
- Sensitive to **outliers** — one extreme value compresses all others
- Preserves the **original distribution shape**

**When to use:**
- Neural Networks (input layer expects [0,1] or [-1,1])
- Image pixel data
- Algorithms that do NOT assume Gaussian distribution

---

### 2️⃣ Standardization (Z-Score Normalization)

Transforms data to have **mean = 0** and **standard deviation = 1**.

**Formula:**

```
X_scaled = (X - μ) / σ

Where:
  μ = mean of the feature
  σ = standard deviation of the feature
```

**Properties:**
- Output range: **unbounded** (typically -3 to +3)
- **Robust to outliers** compared to Min-Max
- Assumes approximately **Gaussian distribution**

**When to use:**
- Logistic Regression, Linear Regression, SVM
- PCA, LDA
- Any algorithm using gradient descent
- Most general-purpose scaling

---

### 3️⃣ Robust Scaling

Uses the **median and IQR (Interquartile Range)** — unaffected by outliers.

**Formula:**

```
X_scaled = (X - median) / IQR

Where:
  IQR = Q3 - Q1  (75th percentile - 25th percentile)
```

**Properties:**
- **Highly robust** to outliers
- Output is **not bounded** to [0,1]
- Centered around 0

**When to use:**
- Data with **significant outliers** you cannot remove
- Financial, medical, sensor data

---

### 4️⃣ MaxAbs Scaling

Scales by the **maximum absolute value** — maps to [-1, 1].

**Formula:**

```
X_scaled = X / |X_max|
```

**Properties:**
- Preserves **sparsity** (zero stays zero)
- Range: **[-1, 1]**
- Does **not shift/center** the data

**When to use:**
- **Sparse data** (text features, TF-IDF matrices)
- Already centered at 0

---

### 5️⃣ Log Transformation

Applies **log** to compress right-skewed distributions.

**Formula:**

```
X_scaled = log(X + 1)     # +1 to handle zeros
```

**Properties:**
- Reduces **right skewness**
- Makes distribution closer to **Gaussian**
- Only valid for **positive values**

**When to use:**
- Highly skewed features (income, population, prices)
- Before applying standardization on skewed data

---

### 6️⃣ Power Transformation (Yeo-Johnson / Box-Cox)

**Box-Cox:** Makes distribution more Gaussian. Requires positive values only.

**Yeo-Johnson:** Extension of Box-Cox — works with **zero and negative values**.

**Formula (simplified):**

```
Box-Cox:    X_scaled = (X^λ - 1) / λ     (λ ≠ 0)
Yeo-Johnson: handles X ≥ 0 and X < 0 separately
```

**When to use:**
- Heavily skewed continuous features
- When you need to satisfy normality assumptions (linear models)

---

### 7️⃣ Unit Vector Scaling (Normalizer)

Scales each **sample (row)** to have unit norm (length = 1).

**Formula (L2 norm):**

```
X_scaled = X / ||X||₂
```

**Properties:**
- Applied **per row** (not per column — unlike all others)
- Useful when **direction matters**, not magnitude

**When to use:**
- Text classification (TF-IDF vectors)
- Cosine similarity computations

---

## 📊 Technique Comparison Table

| Technique | Formula | Range | Outlier Robust | Use Case |
|-----------|---------|-------|:--------------:|----------|
| Min-Max Scaling | (X-min)/(max-min) | [0, 1] | ❌ | Neural nets, no outliers |
| Standardization | (X-μ)/σ | Unbounded | ⚠️ Moderate | General purpose |
| Robust Scaling | (X-median)/IQR | Unbounded | ✅ | Outlier-heavy data |
| MaxAbs Scaling | X/\|max\| | [-1, 1] | ❌ | Sparse data |
| Log Transform | log(X+1) | Varies | ✅ | Right-skewed features |
| Power Transform | Yeo-Johnson/Box-Cox | Varies | ✅ | Heavy skew, normality needed |
| Normalizer | X/\|\|X\|\| | [0, 1] per row | ❌ | Text/cosine similarity |

---

## 🧠 Decision Guide: Which Scaler to Use?

```
Does your data have significant outliers?
  ├── YES → Robust Scaling
  └── NO  → Is the data sparse (many zeros)?
              ├── YES → MaxAbs Scaling
              └── NO  → Is the feature heavily skewed?
                          ├── YES → Log Transform → then Standardize
                          └── NO  → Need bounded [0,1] output?
                                      ├── YES → Min-Max Scaling
                                      └── NO  → Standardization (default)
```

---

## ⚠️ Models That Need vs Don't Need Scaling

### ✅ Must Scale
| Model | Reason |
|-------|--------|
| K-Nearest Neighbors (KNN) | Distance-based |
| Support Vector Machine (SVM) | Margin maximization |
| K-Means Clustering | Euclidean distance |
| Linear / Logistic Regression | Gradient descent |
| Neural Networks | Weight initialization sensitivity |
| PCA / LDA | Variance-based decomposition |
| Ridge / Lasso | Regularization penalty |

### ❌ Scaling NOT Required
| Model | Reason |
|-------|--------|
| Decision Trees | Threshold-based splits |
| Random Forest | Ensemble of trees |
| XGBoost / LightGBM | Tree-based |
| CatBoost | Tree-based |
| Naive Bayes | Probability-based |

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Fitting scaler on full data | Leaks test set statistics | Fit on **train only**, transform both |
| Scaling target variable y | Distorts predictions | Only scale **features X** (except in regression output scaling) |
| Scaling before train-test split | Data leakage | Split first, then scale |
| Using Min-Max with outliers | Extreme compression | Use Robust Scaling instead |
| Scaling tree-based models | Unnecessary computation | Skip for trees |

---

## 🔗 Related Topics

- `Handling_Missing_Values` — Impute nulls **before** scaling
- `Encoding_Categorical_Data` — Encode categories **before** scaling
- `Outlier_Detection` — Handle outliers **before** Min-Max scaling
- `Feature_Engineering` — Create new features, then scale
- `03_Supervised_Learning` — Apply scalers inside ML Pipelines

---

## 📚 References

- Scikit-learn Preprocessing: [https://scikit-learn.org/stable/modules/preprocessing.html](https://scikit-learn.org/stable/modules/preprocessing.html)
- `StandardScaler`: [https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- `MinMaxScaler`: [https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
- `RobustScaler`: [https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)
