# 📘 Train-Test Split — Theory

---

## 📌 What is Train-Test Split?

Train-Test Split is the process of **dividing a dataset into separate subsets** so that a machine learning model is trained on one portion and evaluated on a completely unseen portion — simulating real-world performance.

```
Full Dataset (100%)
        │
        ▼
┌───────────────────────────────────────────┐
│  Training Set (70–80%)  │  Test Set (20–30%) │
│  Model learns here      │  Model evaluated here │
└───────────────────────────────────────────┘
```

> 💡 If you train and evaluate on the same data, your model appears to perform
>    well but fails on new, real-world data — this is called **overfitting**.

---

## 🔍 Why Split the Data?

| Problem | Without Split | With Proper Split |
|---------|--------------|-------------------|
| **Overfitting** | Model memorizes training data ❌ | Detected via train vs test gap ✅ |
| **Generalization** | Unknown real-world performance ❌ | Estimated on held-out test set ✅ |
| **Data Leakage** | Test info leaks into training ❌ | Test set remains completely unseen ✅ |
| **Model Selection** | No unbiased comparison ❌ | Compare models on same test set ✅ |

---

## 🏷️ The Three-Way Split (Train / Validation / Test)

For real ML projects, a **three-way split** is the standard:

```
Full Dataset (100%)
        │
  ┌─────┴──────┐
  │            │
Train (60–70%) Test (15–20%)   ← Never touch until final evaluation
  │
  ├── Train subset  →  Model training
  └── Val subset    →  Hyperparameter tuning & model selection
```

| Split | Purpose |
|-------|---------|
| **Training Set** | Model learns weights/parameters |
| **Validation Set** | Tune hyperparameters, compare models |
| **Test Set** | Final unbiased performance estimate — used ONCE |

> ⚠️ The test set must **never** influence any decision during development.
>    Treat it as a sealed envelope — open it only once at the very end.

---

## 🛠️ Split Strategies

---

### 1️⃣ Simple Random Split

Randomly shuffles the dataset and splits by a fixed ratio.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**When to use:**
- Large datasets (hundreds of rows or more)
- Data is IID (independently and identically distributed)
- No temporal or group structure

**⚠️ Avoid when:**
- Data has time ordering (use Time-Series Split instead)
- Data has group structure (use Group K-Fold instead)

---

### 2️⃣ Stratified Split

Preserves the **class proportion** of the target variable in both train and test sets.

```
Original:  Class A = 80%,  Class B = 20%
After Stratified Split:
  Train  → Class A ≈ 80%,  Class B ≈ 20%  ✅
  Test   → Class A ≈ 80%,  Class B ≈ 20%  ✅
```

**Formula:**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

**When to use:**
- **Classification tasks** — always use stratify=y
- **Imbalanced datasets** — critical for rare class preservation
- Multi-class problems

---

### 3️⃣ K-Fold Cross-Validation

Splits data into **K equal folds**. The model is trained K times, each time using a different fold as validation and the rest as training.

```
K=5 Folds:

Fold 1: [VAL] [TRN] [TRN] [TRN] [TRN]
Fold 2: [TRN] [VAL] [TRN] [TRN] [TRN]
Fold 3: [TRN] [TRN] [VAL] [TRN] [TRN]
Fold 4: [TRN] [TRN] [TRN] [VAL] [TRN]
Fold 5: [TRN] [TRN] [TRN] [TRN] [VAL]

Final Score = mean(score_fold1, ..., score_fold5)
```

**Advantages:**
- Every sample is used for both training and validation
- More **reliable performance estimate** than a single split
- Reduces variance from lucky/unlucky splits

**When to use:**
- Small-to-medium datasets
- More robust model evaluation
- Hyperparameter tuning (with nested CV)

---

### 4️⃣ Stratified K-Fold

K-Fold that **preserves class proportions** in every fold.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
```

**When to use:**
- Classification + K-Fold — always prefer over plain K-Fold

---

### 5️⃣ Time Series Split

Respects **temporal ordering** — training always uses past data, validation uses future data.

```
Split 1: [TRN─────────] [VAL]
Split 2: [TRN──────────────] [VAL]
Split 3: [TRN───────────────────] [VAL]

No future data ever enters training! ✅
```

**When to use:**
- Time-series data (stock prices, sales, weather)
- Any dataset with temporal dependencies
- When shuffling would cause future leakage

---

### 6️⃣ Group K-Fold

Ensures that **no group appears in both train and validation** sets.

```
Groups: [Patient_1, Patient_2, Patient_3, Patient_4, Patient_5]

Fold 1 train: P1, P2, P3, P4 | val: P5
Fold 2 train: P1, P2, P3, P5 | val: P4
...
```

**When to use:**
- Medical data (same patient in both sets = leakage)
- User behavior data (same user in train and test)
- Any grouped or hierarchical data

---

### 7️⃣ Leave-One-Out Cross-Validation (LOOCV)

Extreme case of K-Fold where **K = N** (each sample is its own validation fold).

```
n=5 samples:
Fold 1: [VAL] [TRN] [TRN] [TRN] [TRN]
Fold 2: [TRN] [VAL] [TRN] [TRN] [TRN]
...
Fold 5: [TRN] [TRN] [TRN] [TRN] [VAL]
```

**When to use:**
- Very small datasets (< 50 samples)
- Maximum use of available data

**⚠️ Avoid for:**
- Large datasets — computationally expensive (N model fits)

---

## 📊 Strategy Comparison Table

| Strategy | Data Type | Class Imbalance | Temporal Data | Group Data | Best For |
|----------|-----------|:--------------:|:-------------:|:----------:|----------|
| Random Split | Any | ❌ | ❌ | ❌ | Large, IID data |
| Stratified Split | Classification | ✅ | ❌ | ❌ | Imbalanced classes |
| K-Fold CV | Any | ❌ | ❌ | ❌ | Small-medium data |
| Stratified K-Fold | Classification | ✅ | ❌ | ❌ | **Default for classification** |
| Time Series Split | Time-series | ❌ | ✅ | ❌ | Sequential / temporal data |
| Group K-Fold | Grouped | ❌ | ❌ | ✅ | Patient, user, location data |
| LOOCV | Any | ❌ | ❌ | ❌ | Very small datasets |

---

## 🧠 Decision Guide: Which Strategy to Use?

```
Does your data have temporal ordering?
  ├── YES → Time Series Split
  └── NO  → Does data have group structure (same user/patient/location)?
              ├── YES → Group K-Fold
              └── NO  → Classification or Regression?
                          ├── Classification → Stratified K-Fold (default ✅)
                          └── Regression     → K-Fold CV
                                                 └── Tiny dataset (< 50)?
                                                       ├── YES → LOOCV
                                                       └── NO  → K-Fold (k=5 or k=10)
```

---

## 📐 Choosing the Right Split Ratio

| Dataset Size | Recommended Split | Rationale |
|-------------|------------------|-----------|
| < 1,000 rows | 80/20 or use K-Fold | Preserve as much training data as possible |
| 1,000–10,000 | 80/20 | Standard split |
| 10,000–100,000 | 80/20 or 90/10 | Large test set still gives reliable estimate |
| > 100,000 | 95/5 or 99/1 | Even 1% gives thousands of test samples |

---

## ⚠️ Data Leakage — The #1 Mistake

Data leakage occurs when **information from the test set influences training**.

### Common Sources of Leakage:

| Source | Example | Fix |
|--------|---------|-----|
| **Scaling before split** | StandardScaler fit on full data | Fit on train only |
| **Imputation before split** | Mean imputed from full data | Compute mean from train only |
| **Feature engineering before split** | Group stats from full data | Compute from train only |
| **Target encoding before split** | Mean target from full data | Use K-Fold target encoding |
| **Using future data in time-series** | Next month's data in features | Time Series Split |
| **Sampling before split** | SMOTE on full data | Apply SMOTE after splitting |

```
❌ WRONG:
    scaler.fit_transform(X)        ← learns from entire dataset including test
    X_train, X_test = split(X)

✅ CORRECT:
    X_train, X_test = split(X)
    scaler.fit(X_train)            ← learns from train only
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)  ← applies same learned stats
```

---

## 🔁 Nested Cross-Validation

Used for **unbiased model selection + evaluation** simultaneously.

```
Outer Loop (5 folds) → Performance estimate
  └── Inner Loop (3 folds) → Hyperparameter tuning

For each outer fold:
  Train data → Inner 3-Fold CV → Best hyperparams
  Test fold  → Evaluate with best hyperparams
```

**When to use:**
- Comparing multiple model architectures
- When both tuning and evaluation are needed
- Prevents **selection bias** from using same data for both

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| No stratify on imbalanced data | Test has wrong class ratio | Use `stratify=y` |
| Shuffling time-series data | Future leaks into past | Use `TimeSeriesSplit` |
| Random state not set | Irreproducible results | Always set `random_state=42` |
| Evaluating on train set | Optimistic, misleading score | Always evaluate on held-out set |
| Re-using test set for tuning | Test becomes part of training | Use validation set for tuning |
| Preprocessing before split | Data leakage | Split first, preprocess after |
| Too small test set | High variance estimate | Minimum ~20% or 200+ samples |

---

## 🔗 Related Topics

- `Handling_Missing_Values` — Impute **after** splitting
- `Feature_Scaling` — Scale **after** splitting
- `Feature_Engineering` — Engineer features **after** splitting (for group stats)
- `05_Model_Evaluation` — Cross-validation, bias-variance tradeoff
- `07_Hyperparameter_Tuning` — GridSearchCV uses CV internally

---

## 📚 References

- Scikit-learn Model Selection: [https://scikit-learn.org/stable/modules/cross_validation.html](https://scikit-learn.org/stable/modules/cross_validation.html)
- `train_test_split`: [https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- `StratifiedKFold`: [https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
- `TimeSeriesSplit`: [https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
