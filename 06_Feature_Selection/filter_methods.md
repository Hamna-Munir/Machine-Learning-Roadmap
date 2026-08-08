# 📘 Filter Methods — Feature Selection Theory

---

## 📌 What are Filter Methods?

Filter methods select features based on **statistical properties of the data**  
independently of any machine learning model. They act as a preprocessing step —  
"filtering" out irrelevant or redundant features before training begins.

```
Raw Features (d)     →   Filter Criterion   →   Selected Features (k << d)   →   ML Model
                          (statistical test)
                          No model involved
```

**Key property:** Filter methods are **model-agnostic** — the same selected  
features can be used with any downstream model.

---

## 🔍 Why Feature Selection?

```
Too many features causes:
  ❌ Curse of dimensionality — distances become meaningless in high-d space
  ❌ Overfitting — model memorizes noise features
  ❌ Slow training — more features = more computation
  ❌ Harder interpretation — which features actually matter?
  ❌ Multicollinearity — correlated features confuse some models

Feature selection fixes this by:
  ✅ Keeping only informative, non-redundant features
  ✅ Reducing overfitting risk
  ✅ Speeding up training
  ✅ Improving model interpretability
  ✅ Reducing storage and inference cost
```

---

## 🗂️ Types of Feature Selection Methods

```
Feature Selection
├── Filter Methods          ← This file
│     Rank features by statistical score — independent of model
│     Fast, scalable, model-agnostic
│
├── Wrapper Methods         ← wrapper_methods.md
│     Use model performance to evaluate feature subsets
│     RFE, Forward/Backward selection
│
└── Embedded Methods        ← embedded_methods.md
      Feature selection built into model training
      Lasso (L1), Random Forest importance, XGBoost importance
```

---

## 🧮 Filter Methods — Techniques

---

### 1. Variance Threshold

```
Idea: Remove features with very low variance — they carry little information.
      A feature that is constant (var=0) tells the model nothing.

Threshold:
  var(Xⱼ) < threshold  →  remove feature j

Binary feature special case:
  If a feature is 1 for p% of samples, var = p(1−p)
  threshold=0.80×(1−0.80) = 0.16 removes features that are 1 in >80% of rows

sklearn:
  from sklearn.feature_selection import VarianceThreshold
  sel = VarianceThreshold(threshold=0.01)
  X_filtered = sel.fit_transform(X)

When to use:
  → First step in any pipeline — removes zero/near-zero variance features
  → Especially useful for one-hot encoded features (many near-constant)
  → Fast: O(n × d)
```

---

### 2. Correlation Filter (Pearson)

```
Idea: Remove features that are highly correlated with each other —
      redundant features add noise without adding information.

Steps:
  1. Compute correlation matrix C (d × d)
  2. For each pair (i,j) where |C[i,j]| > threshold:
     Remove the feature with lower correlation to target y

Pearson correlation:
  r(X, y) = Σ(xᵢ − x̄)(yᵢ − ȳ) / (n × σₓ × σᵧ)
  Range: [−1, 1]
  |r| → 1 : strong linear relationship
  |r| → 0 : no linear relationship

Typical threshold: 0.85–0.95

Limitations:
  ❌ Only detects LINEAR relationships
  ❌ Sensitive to outliers
  ❌ Not appropriate for categorical features or classification targets

sklearn:
  No direct class — compute via:
  corr_matrix = pd.DataFrame(X).corr().abs()
  upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
  to_drop = [col for col in upper.columns if any(upper[col] > 0.90)]
```

---

### 3. Chi-Squared Test (χ²)

```
Idea: Test whether a categorical/discrete feature is INDEPENDENT of the target.
      Features with high χ² are strongly associated with the target.

χ² statistic:
  χ² = Σ (O − E)² / E

  Where:
    O = observed frequency
    E = expected frequency under independence

  High χ² → feature and target are NOT independent → informative feature
  Low χ²  → feature and target are independent → remove it

Requirements:
  ✅ Features must be non-negative (counts, frequencies, binary)
  ✅ Target must be categorical (classification only)
  ❌ Not for continuous features or regression

sklearn:
  from sklearn.feature_selection import SelectKBest, chi2
  sel = SelectKBest(score_func=chi2, k=10)
  X_selected = sel.fit_transform(X, y)

  sel.scores_      → χ² score per feature
  sel.pvalues_     → p-value per feature (lower = more significant)
```

---

### 4. ANOVA F-Test (f_classif)

```
Idea: Test whether the MEAN of a continuous feature differs significantly
      across classes. High F-statistic → feature separates classes well.

F-statistic:
  F = Between-group variance / Within-group variance
    = MS_between / MS_within

  High F → class means are far apart relative to within-class spread
           → feature discriminates between classes
  Low F  → class means are similar → feature is uninformative

Requirements:
  ✅ Features are continuous
  ✅ Target is categorical (classification)
  ⚠️ Assumes features are normally distributed within each class
  ⚠️ Assumes equal variance across classes (homoscedasticity)

sklearn:
  from sklearn.feature_selection import SelectKBest, f_classif
  sel = SelectKBest(score_func=f_classif, k=10)
  X_selected = sel.fit_transform(X, y)

  sel.scores_   → F-statistic per feature
  sel.pvalues_  → p-value (lower = more significant)
```

---

### 5. Mutual Information

```
Idea: Measures how much knowing feature Xⱼ reduces uncertainty about target y.
      Captures BOTH linear and non-linear relationships.

MI(X; y) = Σ Σ P(x,y) × log[P(x,y) / (P(x) × P(y))]

MI = 0      → X and y are independent (feature useless)
MI > 0      → X carries information about y (higher = better)
MI = H(y)   → X perfectly predicts y (entropy of target)

Advantages over Pearson / F-test:
  ✅ Detects non-linear relationships
  ✅ Works for both classification and regression
  ✅ No distributional assumptions
  ❌ Slower to compute (uses nearest-neighbor estimation)
  ❌ Requires more data for reliable estimates

sklearn:
  from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

  # Classification
  mi_scores = mutual_info_classif(X, y, random_state=42)

  # Regression
  mi_scores = mutual_info_regression(X, y, random_state=42)

  # Select top k
  sel = SelectKBest(mutual_info_classif, k=10)
  X_selected = sel.fit_transform(X, y)
```

---

### 6. SelectPercentile

```
Instead of selecting a fixed k features, selects the top p% of features
by their score.

from sklearn.feature_selection import SelectPercentile, f_classif
sel = SelectPercentile(f_classif, percentile=25)  # top 25%
X_selected = sel.fit_transform(X, y)

Useful when:
  → You don't know exactly how many features to keep
  → Experimenting with different reduction levels
```

---

### 7. SelectFpr / SelectFdr / SelectFwe (p-value based)

```
SelectFpr: Select features with p-value < alpha (false positive rate control)
SelectFdr: Select features controlling false discovery rate (Benjamini-Hochberg)
SelectFwe: Select features controlling family-wise error (Bonferroni correction)

from sklearn.feature_selection import SelectFpr, f_classif
sel = SelectFpr(f_classif, alpha=0.05)
X_selected = sel.fit_transform(X, y)

When to use:
  → Statistical rigor is required (e.g., scientific reporting)
  → Number of selected features should be driven by significance
```

---

## 📊 Choosing the Right Filter Method

| Method | Feature Type | Target Type | Detects Non-Linear? | Speed |
|--------|:------------:|:-----------:|:-------------------:|:-----:|
| Variance Threshold | Any | Any (unsupervised) | N/A | ✅ Fast |
| Pearson Correlation | Continuous | Continuous | ❌ No | ✅ Fast |
| Chi-Squared | Categorical / Binary | Categorical | ❌ No | ✅ Fast |
| ANOVA F-Test | Continuous | Categorical | ❌ No | ✅ Fast |
| Mutual Information | Any | Any | ✅ Yes | ⚠️ Slower |

---

## ⚙️ SelectKBest — Unified API

```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

# Choose score function based on data type
sel = SelectKBest(score_func=f_classif, k=10)
sel.fit(X_train, y_train)

# Scores and p-values
scores   = sel.scores_
pvalues  = sel.pvalues_

# Transform
X_train_sel = sel.transform(X_train)
X_test_sel  = sel.transform(X_test)   # use same selector!

# Get selected feature names
selected_features = X.columns[sel.get_support()].tolist()
```

---

## 🔄 Filter Methods in a Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler',  StandardScaler()),
    ('select',  SelectKBest(f_classif, k=10)),   # filter step
    ('model',   RandomForestClassifier()),
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

**Critical rule:** Always fit the selector on **training data only** —  
never on the full dataset before cross-validation (data leakage!).

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Fitting selector on full dataset before CV | Data leakage → optimistic scores | Use Pipeline to fit inside CV |
| Using Chi-squared on continuous features | Requires non-negative integers | Use f_classif or MI instead |
| Using Pearson for non-linear relationships | Misses non-linear associations | Use Mutual Information |
| Removing correlated features blindly | May remove an informative feature | Keep the one more correlated with y |
| Not scaling before Variance Threshold | Different units → unfair variance comparison | Scale first or use domain knowledge |
| Selecting k without validation | May select too few or too many | Use CV to choose k |

---

## 🆚 Filter vs Wrapper vs Embedded

| Aspect | Filter | Wrapper | Embedded |
|--------|:------:|:-------:|:--------:|
| Model-agnostic | ✅ Yes | ❌ No | ❌ No |
| Speed | ✅ Fast | ❌ Slow | ✅ Fast |
| Accounts for feature interactions | ❌ No | ✅ Yes | ⚠️ Partial |
| Risk of overfitting selection | Low | High | Low |
| Examples | χ², MI, ANOVA | RFE, Forward | Lasso, RF importance |

---

## 🔗 Related Topics

- `06_Feature_Selection/wrapper_methods.md` — RFE, Forward/Backward selection
- `06_Feature_Selection/embedded_methods.md` — Lasso, RF importance
- `05_Model_Evaluation/cross_validation.md` — Use CV to validate selected features
- `04_Unsupervised_Learning/PCA` — Dimensionality reduction vs feature selection
- `Lasso_Regression` — L1 regularization as embedded feature selector

---

## 📚 References

- Scikit-learn Feature Selection: [https://scikit-learn.org/stable/modules/feature_selection.html](https://scikit-learn.org/stable/modules/feature_selection.html)
- An Introduction to Statistical Learning — Chapter 6.1 (Subset Selection)
- The Elements of Statistical Learning — Chapter 3.3
