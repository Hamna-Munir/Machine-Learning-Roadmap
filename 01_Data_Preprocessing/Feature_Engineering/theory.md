# 📘 Feature Engineering — Theory

---

## 📌 What is Feature Engineering?

Feature Engineering is the process of using **domain knowledge and data transformation techniques** to create, modify, or select features that make machine learning models more accurate and efficient.

```
Raw Data  →  Feature Engineering  →  Better Features  →  Better Model
```

> 💡 "Feature engineering is the art of turning raw data into representations
>     that ML algorithms can learn from effectively."

---

## 🔍 Why Does Feature Engineering Matter?

| Without Feature Engineering | With Feature Engineering |
|-----------------------------|--------------------------|
| Model sees: Date = "2024-01-15" | Model sees: Year=2024, Month=1, DayOfWeek=1, IsWeekend=0 |
| Model sees: Address = "New York" | Model sees: Latitude=40.71, Longitude=-74.00 |
| Model sees: Name = "John Smith" | Model sees: NameLength=10, HasMiddleName=0 |
| Weak patterns, low accuracy ❌ | Rich patterns, higher accuracy ✅ |

---

## 🏷️ Types of Feature Engineering

```
Feature Engineering
│
├── 1. Feature Creation        → Generate new features from existing ones
├── 2. Feature Transformation  → Change distribution or scale of features
├── 3. Feature Interaction     → Combine features to capture relationships
├── 4. Date / Time Features    → Extract temporal patterns
├── 5. Text Features           → Convert text to numeric representations
├── 6. Aggregation Features    → Group-level statistics
├── 7. Binning / Discretization→ Convert continuous to categorical
└── 8. Domain-Specific         → Use business knowledge to craft features
```

---

## 🛠️ Techniques

---

### 1️⃣ Feature Creation

Creating **entirely new features** from existing columns using mathematical operations or domain logic.

**Examples:**

```python
# From Age and Salary
df['Salary_Per_Age'] = df['Salary'] / df['Age']

# From height and weight
df['BMI'] = df['Weight'] / (df['Height'] ** 2)

# From purchase history
df['Avg_Order_Value'] = df['Total_Revenue'] / df['Num_Orders']
```

**When to use:**
- When domain knowledge suggests a derived metric is meaningful
- When raw features alone are insufficient for the model

---

### 2️⃣ Feature Transformation

Changing the **distribution or scale** of an existing feature.

| Transformation | Formula | Purpose |
|---------------|---------|---------|
| Log Transform | `log(X + 1)` | Reduce right skew |
| Square Root | `√X` | Moderate right skew |
| Square | `X²` | Capture non-linear patterns |
| Reciprocal | `1/X` | Inverse relationships |
| Box-Cox / Yeo-Johnson | Auto-λ | Maximize normality |

**When to use:**
- Feature is heavily skewed
- Model assumes normality (Linear Regression)
- Non-linear relationship with target

---

### 3️⃣ Polynomial & Interaction Features

Creating **products and powers** of features to capture non-linear relationships.

**Formula:**
```
Degree-2 polynomial of [X₁, X₂]:
→ [1, X₁, X₂, X₁², X₁·X₂, X₂²]
```

**Examples:**
```python
# Manual interaction
df['Age_x_Salary']   = df['Age'] * df['Salary']
df['Age_squared']    = df['Age'] ** 2

# sklearn PolynomialFeatures (degree=2)
# [a, b] → [1, a, b, a², ab, b²]
```

**⚠️ Warning:**
- `n` features at degree `d` → `C(n+d, d)` new features
- Degree 2 with 10 features → 66 features — can cause **dimensionality explosion**
- Use `interaction_only=True` to limit to cross terms only

---

### 4️⃣ Date & Time Features

Extracting **temporal signals** from datetime columns.

```python
df['Year']        = df['Date'].dt.year
df['Month']       = df['Date'].dt.month
df['Day']         = df['Date'].dt.day
df['DayOfWeek']   = df['Date'].dt.dayofweek   # 0=Mon, 6=Sun
df['IsWeekend']   = df['Date'].dt.dayofweek >= 5
df['Quarter']     = df['Date'].dt.quarter
df['WeekOfYear']  = df['Date'].dt.isocalendar().week
df['IsMonthStart']= df['Date'].dt.is_month_start
df['Hour']        = df['Date'].dt.hour         # for timestamps
df['Days_Since']  = (pd.Timestamp.now() - df['Date']).dt.days
```

**Cyclic Encoding for Time (sine/cosine):**
```
Month is cyclic: January (1) is close to December (12)
→ Encode with sin/cos to capture this:

df['Month_sin'] = sin(2π × Month / 12)
df['Month_cos'] = cos(2π × Month / 12)
```

**When to use:**
- Any column containing dates or timestamps
- Time-series data, sales data, log data

---

### 5️⃣ Text-Based Features

Converting **string columns** into numeric signals.

```python
# Length-based
df['Name_Length']      = df['Name'].str.len()
df['Word_Count']       = df['Description'].str.split().str.len()

# Pattern-based
df['Has_Email']        = df['Contact'].str.contains('@').astype(int)
df['Num_Digits']       = df['Text'].str.count(r'\d')
df['Has_Special_Char'] = df['Text'].str.contains(r'[^a-zA-Z0-9]').astype(int)

# Case-based
df['Is_Uppercase']     = df['Text'].str.isupper().astype(int)

# Extraction
df['Domain']           = df['Email'].str.split('@').str[1]
```

**When to use:**
- Name, address, description, comment columns
- Before applying full NLP (TF-IDF, embeddings)

---

### 6️⃣ Aggregation / Group-Level Features

Creating **group statistics** that capture context for each row.

```python
# Group aggregations
group = df.groupby('City')['Salary']

df['City_Mean_Salary']   = df['City'].map(group.mean())
df['City_Median_Salary'] = df['City'].map(group.median())
df['City_Max_Salary']    = df['City'].map(group.max())
df['City_Std_Salary']    = df['City'].map(group.std())
df['City_Count']         = df['City'].map(group.count())

# Deviation from group mean
df['Salary_vs_City_Mean'] = df['Salary'] - df['City_Mean_Salary']
```

**When to use:**
- Categorical column + numeric target relationship
- Customer/city/product-level statistics in transactional data
- ⚠️ Compute on **train set only** to avoid leakage

---

### 7️⃣ Binning / Discretization

Converting **continuous features into discrete bins** (categories).

**Equal-width binning:**
```python
df['Age_Bin'] = pd.cut(df['Age'], bins=5, labels=['Teen','Young','Adult','Middle','Senior'])
```

**Equal-frequency (quantile) binning:**
```python
df['Salary_Quartile'] = pd.qcut(df['Salary'], q=4, labels=['Low','Mid','High','VeryHigh'])
```

**Custom thresholds (domain knowledge):**
```python
bins   = [0, 18, 35, 60, 100]
labels = ['Minor', 'Young Adult', 'Adult', 'Senior']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
```

**When to use:**
- Feature has non-linear relationship with target
- Reduce impact of outliers within bins
- Domain naturally defines categories (age groups, income brackets)

---

### 8️⃣ Flag / Indicator Features

Creating **binary flags** from conditions in the data.

```python
df['Is_High_Earner']   = (df['Salary'] > 100_000).astype(int)
df['Is_Senior']        = (df['Age'] > 60).astype(int)
df['Has_Experience']   = (df['Years_Exp'] > 5).astype(int)
df['Is_New_Customer']  = (df['Tenure_Days'] < 30).astype(int)
df['Is_Missing_Email'] = df['Email'].isna().astype(int)
```

**When to use:**
- Domain-defined thresholds exist
- Missing value indicator (captures missingness as a pattern)
- Binary events (is_weekend, is_holiday, is_promoted)

---

### 9️⃣ Ratio & Difference Features

Capturing **relative relationships** between two features.

```python
# Ratios
df['Expense_Ratio']       = df['Expenses'] / df['Income']
df['Click_Through_Rate']  = df['Clicks']   / df['Impressions']
df['Profit_Margin']       = df['Profit']   / df['Revenue']

# Differences
df['Age_at_Hire']         = df['Hire_Year']  - df['Birth_Year']
df['Days_to_Delivery']    = df['Delivered']  - df['Ordered']
df['Score_Improvement']   = df['Score_Post'] - df['Score_Pre']
```

---

### 🔟 Domain-Specific Features

Features crafted using **business or scientific knowledge**.

| Domain | Raw Feature | Engineered Feature |
|--------|-------------|-------------------|
| E-commerce | order_date, ship_date | days_to_ship |
| Finance | revenue, expenses | profit_margin |
| Healthcare | height, weight | BMI |
| HR | hire_date, today | tenure_days |
| Marketing | clicks, impressions | click_through_rate |
| Real Estate | price, sqft | price_per_sqft |

---

## 📊 Feature Engineering Pipeline Order

```
Raw Data
    │
    ▼
Handle Missing Values          (impute or flag)
    │
    ▼
Encode Categorical Features    (OHE / Ordinal / Target)
    │
    ▼
Create New Features            (ratios, flags, interactions)
    │
    ▼
Extract Date/Time Features     (year, month, dayofweek)
    │
    ▼
Apply Transformations          (log, sqrt, polynomial)
    │
    ▼
Aggregate Group Features       (mean/std by group)
    │
    ▼
Binning / Discretization       (if needed)
    │
    ▼
Scale Features                 (StandardScaler / MinMax)
    │
    ▼
Feature Selection              (drop redundant/low-importance)
    │
    ▼
Train Model
```

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Creating features from full data | Data leakage | Compute group stats on train only |
| Too many polynomial features | Curse of dimensionality | Limit degree, use `interaction_only` |
| Ignoring datetime columns | Lose temporal patterns | Always extract dt features |
| Skipping domain knowledge | Generic features | Consult domain experts |
| Not validating new features | Noise added | Check correlation with target |
| Engineering before split | Leakage from test set | Split first, engineer after |

---

## 🔗 Related Topics

- `Handling_Missing_Values` — Handle nulls before engineering
- `Encoding_Categorical_Data` — Encode after creating new categorical bins
- `Feature_Scaling` — Scale engineered numeric features
- `Outlier_Detection` — Handle outliers before ratio/log features
- `06_Feature_Selection` — Select best engineered features

---

## 📚 References

- Scikit-learn `PolynomialFeatures`: [https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
- Pandas `dt` accessor: [https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.html](https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.html)
- Feature Engineering for ML (Book): Casari & Zheng — O'Reilly
- Kaggle Feature Engineering Course: [https://www.kaggle.com/learn/feature-engineering](https://www.kaggle.com/learn/feature-engineering)
