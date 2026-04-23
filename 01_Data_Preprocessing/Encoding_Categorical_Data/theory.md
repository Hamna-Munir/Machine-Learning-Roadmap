# 📘 Encoding Categorical Data — Theory

---

## 📌 What is Categorical Data?

Categorical data represents **discrete groups or categories** rather than continuous numeric values.

| Type | Description | Example |
|------|-------------|---------|
| **Nominal** | No inherent order | Color: Red, Blue, Green |
| **Ordinal** | Has a meaningful order | Education: High School < Bachelor's < Master's |
| **Binary** | Only two categories | Gender: Male / Female |

> ⚠️ Most ML algorithms require **numeric input**. Categorical data must be encoded before training.

---

## 🔠 Why Encoding Matters

- Machine learning models work with **numbers, not strings**
- Wrong encoding can introduce **false ordinal relationships**
- Proper encoding can significantly affect **model accuracy**

---

## 🛠️ Encoding Techniques

---

### 1️⃣ Label Encoding

Assigns an **integer** to each category.

```
Red   → 0
Blue  → 1
Green → 2
```

**When to use:**
- Ordinal data (Low → Medium → High)
- Tree-based models (Decision Tree, Random Forest, XGBoost)

**⚠️ Avoid for:**
- Nominal data with linear models — implies false ordering (Red < Blue < Green)

---

### 2️⃣ One-Hot Encoding (OHE)

Creates a **binary column for each category**.

```
Color    →   Red  Blue  Green
Red      →    1    0     0
Blue     →    0    1     0
Green    →    0    0     1
```

**When to use:**
- Nominal categories
- Linear models, Logistic Regression, SVMs, Neural Networks

**⚠️ Watch out for:**
- **Dummy Variable Trap** → Drop one column to avoid multicollinearity
- **High cardinality** → Too many unique values = too many columns (curse of dimensionality)

---

### 3️⃣ Ordinal Encoding

Manually maps **ordered categories** to integers that preserve rank.

```
Education:
  High School  → 1
  Bachelor's   → 2
  Master's     → 3
  PhD          → 4
```

**When to use:**
- Explicitly ordinal features
- When order must be preserved for the model to learn correctly

---

### 4️⃣ Binary Encoding

Converts categories to **integer → binary representation** → each bit becomes a column.

```
Cat ID → Binary → Columns
1      →  001   → [0, 0, 1]
2      →  010   → [0, 1, 0]
3      →  011   → [0, 1, 1]
```

**When to use:**
- High cardinality features
- More compact than OHE

---

### 5️⃣ Frequency / Count Encoding

Replaces categories with their **frequency (count)** in the dataset.

```
City     Count   → Encoded
London   500     → 500
Paris    300     → 300
Berlin   200     → 200
```

**When to use:**
- High cardinality features
- Frequency is meaningful for the target variable

---

### 6️⃣ Target Encoding (Mean Encoding)

Replaces each category with the **mean of the target variable** for that category.

```
City    Avg(House Price)  → Encoded
London  450,000           → 450000
Paris   320,000           → 320000
```

**When to use:**
- High cardinality with strong category-target relationship

**⚠️ Risk:**
- **Data leakage** — always apply within cross-validation folds

---

### 7️⃣ Hashing Encoding (Feature Hashing)

Uses a **hash function** to map categories to a fixed number of columns.

**When to use:**
- Very high cardinality
- Online learning / streaming data

---

## 📊 Technique Comparison Table

| Technique | Ordinal Support | High Cardinality | Risk of Leakage | Adds Columns |
|-----------|:--------------:|:----------------:|:---------------:|:------------:|
| Label Encoding | ✅ | ✅ | ❌ | ❌ |
| One-Hot Encoding | ❌ | ❌ | ❌ | ✅ |
| Ordinal Encoding | ✅ | ✅ | ❌ | ❌ |
| Binary Encoding | ❌ | ✅ | ❌ | Partial |
| Frequency Encoding | ❌ | ✅ | ❌ | ❌ |
| Target Encoding | ❌ | ✅ | ✅ | ❌ |
| Hashing Encoding | ❌ | ✅ | ❌ | Fixed |

---

## 🧠 Decision Guide: Which Encoding to Use?

```
Is the feature ordinal?
  ├── YES → Ordinal Encoding
  └── NO  → How many unique values?
              ├── Few (< 10)  → One-Hot Encoding
              ├── Many (≥ 10) → Binary / Frequency / Target Encoding
              └── Huge (100+) → Hashing Encoding
```

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Label Encoding on nominal data | Introduces false order | Use OHE instead |
| OHE on high cardinality | Dimensionality explosion | Use Binary or Target Encoding |
| Target Encoding without CV | Data leakage | Apply inside CV folds |
| Forgetting to drop one OHE column | Multicollinearity (dummy trap) | Use `drop_first=True` |
| Applying encoding before train-test split | Leakage from test set | Fit encoder on train only |

---

## 🔗 Related Topics

- `Handling_Missing_Values` — Handle nulls **before** encoding
- `Feature_Scaling` — Scale numeric-encoded features **after** encoding
- `Feature_Engineering` — Create new features from encoded categories
- `Feature_Selection` — Drop low-variance or redundant encoded columns

---

## 📚 References

- Scikit-learn: [Preprocessing Categorical Features](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features)
- Category Encoders Library: [https://contrib.scikit-learn.org/category_encoders/](https://contrib.scikit-learn.org/category_encoders/)
- Pandas `get_dummies()`: [https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)
