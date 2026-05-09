# 📘 Correlation Analysis — Theory

---

## 📌 What is Correlation Analysis?

Correlation Analysis is the **systematic measurement of the strength and direction**  
of relationships between variables — across all pairs in a dataset simultaneously.

```
Bivariate Analysis    →  Examines one pair at a time
Correlation Analysis  →  Examines ALL pairs systematically
                         + ranks features by predictive power
                         + detects multicollinearity
                         + guides feature selection
```

> 💡 "Correlation analysis transforms intuition into numbers —  
>      it tells you not just whether features are related,  
>      but how strongly, in what direction, and whether to trust the measure."

---

## 🔍 Why Correlation Analysis?

| Goal | What It Reveals |
|------|----------------|
| **Feature-target ranking** | Which features most strongly predict the outcome? |
| **Redundancy detection** | Are two features capturing the same information? |
| **Multicollinearity check** | Which features will hurt linear model performance? |
| **Feature selection** | Which features to keep, merge, or drop? |
| **Model assumption check** | Do features satisfy independence assumptions? |
| **Data understanding** | What hidden structure exists in the data? |

---

## 🏷️ Types of Correlation Coefficients

```
Correlation Coefficients
    │
    ├── Pearson r          → Linear, continuous, parametric
    ├── Spearman ρ         → Monotonic, ordinal/non-normal, non-parametric
    ├── Kendall's τ        → Monotonic, small samples, concordance-based
    ├── Point-Biserial rpb → Continuous vs binary
    ├── Phi Coefficient φ  → Binary vs binary
    └── Cramér's V         → Categorical vs categorical (any size)
```

---

## 🛠️ Correlation Techniques

---

### 1️⃣ Pearson Correlation (r)

Measures the **linear** relationship between two continuous variables.

**Formula:**
```
        Σ(xᵢ − x̄)(yᵢ − ȳ)
r = ─────────────────────────────
     √[Σ(xᵢ − x̄)²] × √[Σ(yᵢ − ȳ)²]
```

**Interpretation Scale:**
```
|r| = 1.00          → Perfect linear relationship
|r| = 0.80 – 0.99   → Very strong
|r| = 0.60 – 0.79   → Strong
|r| = 0.40 – 0.59   → Moderate
|r| = 0.20 – 0.39   → Weak
|r| = 0.00 – 0.19   → Negligible / No relationship

Positive r → both variables increase together
Negative r → one increases as the other decreases
```

**Assumptions:**
- Both variables are **continuous**
- Relationship is **linear**
- Both variables are approximately **normally distributed**
- No significant **outliers** (they distort r heavily)

**⚠️ Key Limitations:**
- Misses **non-linear** relationships completely
- Sensitive to **outliers** — one extreme point can flip r from +0.8 to −0.3
- Does **NOT** imply causation
- Spurious correlations exist in large datasets

---

### 2️⃣ Spearman Rank Correlation (ρ)

A **non-parametric** alternative — converts values to ranks, then applies Pearson  
to the ranks. Measures **monotonic** (consistently increasing or decreasing) relationships.

**Formula:**
```
ρ = 1 − (6 × Σdᵢ²) / (n × (n² − 1))

Where dᵢ = rank(xᵢ) − rank(yᵢ)
```

**When Spearman > Pearson (use Spearman instead):**
- Non-normal distributions
- Ordinal variables
- Outliers present
- Monotonic but non-linear relationship

**Interpretation:** Same scale as Pearson (−1 to +1).

```
Pearson r  = 0.25   (misses the real pattern)
Spearman ρ = 0.78   → Strong monotonic relationship ✅
                       The relationship is non-linear!
```

---

### 3️⃣ Kendall's Tau (τ)

Measures the **concordance** between two rankings — the proportion of pairs that are  
ordered consistently in both variables.

**Formula:**
```
τ = (Concordant Pairs − Discordant Pairs) / (n × (n−1) / 2)

Concordant pair: xᵢ > xⱼ AND yᵢ > yⱼ  (same order in both)
Discordant pair: xᵢ > xⱼ AND yᵢ < yⱼ  (opposite order)
```

**When to use over Spearman:**
- Very **small samples** (N < 30)
- Many **tied ranks**
- Need a statistic with clearer probabilistic interpretation

**Interpretation:**
```
τ = +1  → Perfect concordance (all pairs agree in order)
τ =  0  → No relationship
τ = −1  → Perfect discordance (all pairs disagree in order)
```

**Note:** Kendall τ values are typically smaller than Spearman ρ for the same data —  
this is expected and does NOT mean the relationship is weaker.

---

### 4️⃣ Point-Biserial Correlation (rpb)

Measures the relationship between a **continuous variable** and a **binary (0/1) variable**.

**Formula:**
```
        (M₁ − M₀)       √(n₁ × n₀)
rpb = ────────────── × ──────────────
           s                 n

Where:
  M₁, M₀ = mean of continuous variable for group 1 and 0
  n₁, n₀ = number in each group
  s       = standard deviation of continuous variable
  n       = total sample size
```

**Interpretation:** Same scale as Pearson (−1 to +1).

**When to use:**
- Evaluating each continuous feature's relationship with a binary target
- Selecting features for binary classification models

---

### 5️⃣ Phi Coefficient (φ)

Measures association between **two binary variables**.

**Formula:**
```
         (ad − bc)
φ = ─────────────────────────────────
     √[(a+b)(c+d)(a+c)(b+d)]

Using the 2×2 contingency table:
        Y=0    Y=1
X=0  [  a   |  b  ]
X=1  [  c   |  d  ]
```

**Interpretation:** Same scale as Pearson (−1 to +1).

---

### 6️⃣ Cramér's V

Measures association between **two categorical variables** of any size  
(generalization of Phi to k×k tables).

**Formula:**
```
V = √[χ² / (n × min(r−1, c−1))]

Where:
  χ²  = chi-square statistic
  n   = total observations
  r,c = number of rows and columns
```

**Interpretation:**
```
V = 0.00–0.10  → Negligible association
V = 0.10–0.30  → Weak association
V = 0.30–0.50  → Moderate association
V = 0.50–1.00  → Strong association
```

---

## 📊 Correlation Matrix

The **correlation matrix** shows all pairwise correlations simultaneously —  
an n × n symmetric matrix where each cell is the correlation between two features.

```
         Age  Salary  Score  Exp
Age     [ 1.0   0.65   0.42  0.80]
Salary  [ 0.65  1.0    0.70  0.55]
Score   [ 0.42  0.70   1.0   0.35]
Exp     [ 0.80  0.55   0.35  1.0 ]

Diagonal = 1.0 (every feature perfectly correlates with itself)
Matrix is symmetric (r(A,B) = r(B,A))
```

**Visualized as a heatmap:**
- Dark red → strong positive correlation
- Dark blue → strong negative correlation
- White/light → near-zero correlation

---

## 🛠️ Correlation Heatmap Variants

| Variant | When to Use |
|---------|------------|
| **Full matrix** | Complete overview — symmetric |
| **Lower triangle** | Cleaner — removes duplicate information |
| **Masked diagonals** | Remove self-correlations |
| **Clustered heatmap** | Groups correlated features together (hierarchical clustering on rows/cols) |
| **Absolute value heatmap** | When direction doesn't matter — only strength |

---

## 📐 Statistical Significance of Correlation

A correlation coefficient alone is not enough — we need to know if it is **statistically significant** (not due to chance).

**Hypothesis test:**
```
H₀: ρ = 0  (no linear relationship in the population)
H₁: ρ ≠ 0  (a relationship exists)

t-statistic: t = r × √(n−2) / √(1−r²)
Degrees of freedom: df = n − 2

p-value < 0.05  →  Reject H₀ → Correlation is significant ✅
p-value ≥ 0.05  →  Fail to reject H₀ → May be due to chance ❌
```

**⚠️ Important caveat:**
- In large samples (n > 1000), even **tiny correlations** (r = 0.05) become statistically significant
- Always report **both p-value and effect size (r)** — statistical significance ≠ practical significance

---

## 🧠 Correlation vs Causation

```
Correlation:   Ice cream sales ↑  AND  Drowning deaths ↑
               r = 0.85 (very strong!)

Causation?     NO — both are driven by a confounding variable: HOT WEATHER

Correlation tells you WHAT. It never tells you WHY.
```

**Three possible explanations for X correlates with Y:**
```
1. X causes Y            (direct causation)
2. Y causes X            (reverse causation)
3. Z causes both X and Y (confounding variable)
```

---

## 🛠️ Multicollinearity in Feature Space

Multicollinearity occurs when **two or more features are highly correlated** with each other —  
meaning they carry redundant information.

**Problems it causes:**
- Unstable regression coefficients (high variance)
- Difficult to interpret which feature is truly important
- Inflated standard errors → misleading p-values

**Detection methods:**
```
1. Correlation matrix         → |r| > 0.85 between two features
2. VIF (Variance Inflation)  → VIF > 10 for any feature
3. Eigenvalues of corr matrix → Near-zero eigenvalue = collinearity
```

**Solutions:**
```
|r| > 0.85  →  Drop one of the two features
|r| > 0.85  →  Create a combined feature (ratio, difference, PCA)
VIF > 10    →  Drop or regularize (Ridge regression)
```

---

## 🧠 Decision Guide: Which Correlation to Use?

```
What types of variables do you have?
    │
    ├── Both CONTINUOUS / NUMERICAL
    │       ├── Is the relationship expected to be LINEAR?
    │       │       ├── YES → Pearson r
    │       │       └── NO  → Spearman ρ
    │       ├── Are there OUTLIERS or non-normal distributions?
    │       │       └── YES → Spearman ρ (robust)
    │       └── Small sample (N < 30) with tied ranks?
    │               └── YES → Kendall's τ
    │
    ├── CONTINUOUS vs BINARY (0/1)
    │       └── Point-Biserial rpb
    │
    ├── BINARY vs BINARY
    │       └── Phi Coefficient φ
    │
    └── CATEGORICAL vs CATEGORICAL (any size)
            └── Cramér's V (via Chi-Square)
```

---

## 📊 Correlation Coefficient Comparison Table

| Method | Variable Types | Linearity | Parametric | Outlier Robust |
|--------|:-------------:|:---------:|:----------:|:--------------:|
| Pearson r | Num × Num | Linear | ✅ Yes | ❌ No |
| Spearman ρ | Num × Num | Monotonic | ❌ No | ✅ Yes |
| Kendall τ | Num × Num | Monotonic | ❌ No | ✅ Yes |
| Point-Biserial | Num × Binary | Linear | ✅ Yes | ❌ No |
| Phi φ | Binary × Binary | — | ✅ Yes | — |
| Cramér's V | Cat × Cat | — | ❌ No | — |

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Using Pearson on non-linear data | r ≈ 0 even for strong curves | Plot first; use Spearman |
| Ignoring outliers before Pearson | Outliers distort r severely | Check scatter before trusting r |
| Treating significance as strength | n=10,000 → r=0.05 is significant | Always report both r AND p-value |
| Assuming correlation = causation | May be a confounding variable | Domain knowledge required |
| High feature-feature r → keep both | Multicollinearity hurts linear models | Drop one or apply PCA/VIF |
| Using Pearson on ordinal variables | Interval arithmetic on ranks is wrong | Use Spearman or Kendall instead |
| Not checking the correlation sign | Negative r flips interpretation | Direction matters as much as magnitude |

---

## 🔗 Related Topics

- `Bivariate_Analysis` — Scatter plots and tests for individual feature pairs
- `Multivariate_Analysis` — Correlation matrix + VIF + PCA in full context
- `06_Feature_Selection` — Use correlation to remove redundant features
- `07_Hyperparameter_Tuning` — Regularization addresses multicollinearity
- `03_Supervised_Learning/Ridge_Regression` — Handles correlated features via L2 penalty

---

## 📚 References

- SciPy `pearsonr`, `spearmanr`, `kendalltau`: [https://docs.scipy.org/doc/scipy/reference/stats.html](https://docs.scipy.org/doc/scipy/reference/stats.html)
- Pandas `DataFrame.corr()`: [https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html)
- Seaborn Heatmap: [https://seaborn.pydata.org/generated/seaborn.heatmap.html](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
- Statsmodels VIF: [https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html](https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html)
- Cramér's V: [https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V](https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V)
