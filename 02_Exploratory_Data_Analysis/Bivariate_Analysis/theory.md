# 📘 Bivariate Analysis — Theory

---

## 📌 What is Bivariate Analysis?

Bivariate Analysis is the process of analyzing **two variables simultaneously** to understand the relationship, association, or dependency between them.

```
Univariate  →  One variable at a time   (What does X look like?)
Bivariate   →  Two variables together   (How does X relate to Y?)
Multivariate→  Three or more variables  (How do X, Y, Z interact?)
```

> 💡 "Univariate tells you what each feature looks like.  
>      Bivariate tells you how features talk to each other."

---

## 🔍 Why Bivariate Analysis?

| Goal | What It Reveals |
|------|----------------|
| **Feature-target relationship** | Which features predict the target? |
| **Feature-feature correlation** | Are two features redundant? |
| **Group differences** | Do different categories have different distributions? |
| **Linear vs non-linear** | What type of relationship exists? |
| **Outlier context** | Is a point an outlier in both X and Y? |
| **Multicollinearity signal** | Are two predictors highly correlated? |

---

## 🏷️ Variable Pair Types

The technique you use depends on the **types of the two variables** being analyzed:

```
Variable Pair
    │
    ├── Numerical   vs Numerical    →  Scatter plot, Correlation, Line plot
    ├── Categorical vs Numerical    →  Box plot, Violin plot, Bar chart (mean)
    └── Categorical vs Categorical  →  Cross-tab, Heatmap, Stacked bar, Chi-Square
```

---

## 🛠️ Case 1: Numerical vs Numerical

---

### 1️⃣ Scatter Plot

Plots one numerical variable on the X-axis and another on the Y-axis.  
Each point represents **one observation**.

```
Y ▲
  │       ●  ●
  │     ●   ●  ●
  │   ●  ●
  │ ●  ●
  └──────────────► X

Positive linear relationship → points trend upward ↗
```

**What to look for:**
- **Direction** → positive (↗) or negative (↘) trend
- **Form** → linear or curved (non-linear)
- **Strength** → tight cluster (strong) vs scattered (weak)
- **Outliers** → isolated points far from the main cluster

**When to use:**
- Both variables are continuous
- Checking linearity assumption before regression
- Detecting multicollinearity between two features

---

### 2️⃣ Regression Line (Line of Best Fit)

A line drawn through the scatter plot that **minimizes residuals**.

```
Y ▲       ●  ●
  │     ●/●  ●
  │   ●/ ●
  │ ●/●        ← regression line
  └──────────────► X
```

**Formula:**
```
ŷ = β₀ + β₁x

β₁ = Σ[(x - x̄)(y - ȳ)] / Σ[(x - x̄)²]   (slope)
β₀ = ȳ - β₁x̄                              (intercept)
```

---

### 3️⃣ Pearson Correlation Coefficient (r)

Measures the **strength and direction** of a linear relationship between two numerical variables.

**Formula:**
```
r = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / [n × σₓ × σᵧ]

Where:
  x̄, ȳ = means of X and Y
  σₓ, σᵧ = standard deviations of X and Y
```

**Interpretation:**
```
r = +1.0    Perfect positive linear relationship
r =  0.7    Strong positive
r =  0.4    Moderate positive
r =  0.0    No linear relationship
r = -0.4    Moderate negative
r = -0.7    Strong negative
r = -1.0    Perfect negative linear relationship
```

**Visual:**
```
r ≈ +1          r ≈ 0           r ≈ -1
  ●●             ●  ●              ●
  ●●●          ●  ● ●           ●●●
  ●●●        ●●  ●●          ●●●●
  ●●         ●●●           ●●●
```

**Assumptions:**
- Both variables are continuous
- Relationship is **linear**
- No severe outliers (Pearson is sensitive)

**⚠️ Limitation:** Pearson only captures **linear** relationships — it will miss curved patterns.

---

### 4️⃣ Spearman Rank Correlation (ρ)

A **non-parametric** version of Pearson — measures monotonic relationship (not necessarily linear).

**Formula:**
```
ρ = 1 - (6 × Σdᵢ²) / (n(n² - 1))

Where dᵢ = difference in ranks of xᵢ and yᵢ
```

**When to use instead of Pearson:**
- Data is **ordinal** (ranked categories)
- Relationship is **non-linear but monotonic**
- Data contains **outliers** (Spearman is robust)
- Variables are **not normally distributed**

---

### 5️⃣ Correlation vs Causation

```
⚠️ High correlation does NOT mean causation!

Example:
  Ice cream sales ↑ and drowning incidents ↑  →  r = 0.85
  BUT ice cream does NOT cause drowning
  Both are caused by a third variable: HOT WEATHER (confounding variable)
```

---

### 6️⃣ Joint Plot

Combines a **scatter plot** in the center with **marginal histograms or KDE plots** on the axes.

```
     │▐█                         marginal KDE of Y
     │▐███
     │▐█████
     ├────────────────────────
     │  ●   ●
     │    ●   ●  ●   scatter plot
     │  ●  ●  ●
     ├────────────────────────
         ████████               marginal histogram of X
```

---

### 7️⃣ Hexbin Plot

An alternative to scatter plots for **large datasets** — bins points into hexagonal cells colored by count.

**When to use:**
- N > 10,000 points (scatter overplotting becomes unreadable)
- Want to see density of points simultaneously

---

## 🛠️ Case 2: Categorical vs Numerical

---

### 8️⃣ Box Plot by Group

Displays the **distribution of a numerical variable** across different categories.

```
Category A:    ──[════|════]──       ●
Category B:       ──[══|════════]──
Category C:   ──[═══|══]──   ●  ●
                Q1  Med Q3
```

**What to look for:**
- Are medians different across groups?
- Does spread (IQR) differ across groups?
- Do any groups have outliers?

---

### 9️⃣ Violin Plot by Group

Box plot + KDE shape per group — shows **full distribution per category**.

**When to use:**
- When knowing the shape (unimodal, bimodal) of each group matters

---

### 🔟 Bar Chart of Group Means (with Error Bars)

Shows the **mean value** of a numerical variable for each category, with error bars for confidence.

```
Mean Value
    │   ████
    │   ████±     ████
    │   ████   ████±     ████
    │   ████   ████   ████±
    └──────────────────────────── Categories
         A       B       C
```

**⚠️ Limitation:** Bar charts hide distribution shape — always pair with a box or violin plot.

---

### 1️⃣1️⃣ Strip Plot / Swarm Plot

Shows **every individual data point** grouped by category.

```
Category A:    ● ●  ● ● ●  ● ●
Category B:  ● ● ● ●  ●●● ●
Category C:       ●  ● ●  ●● ●  ●
```

**When to use:**
- Small to medium datasets (N < 500)
- Want to see actual data points instead of summaries

---

### 1️⃣2️⃣ Point Plot

Displays **mean ± confidence interval** per category connected by lines.

**When to use:**
- Showing trends across ordered categories
- When error bars (uncertainty) are important to show

---

## 🛠️ Case 3: Categorical vs Categorical

---

### 1️⃣3️⃣ Cross-Tabulation (Contingency Table)

A frequency table showing the **count or proportion** of combinations of two categorical variables.

```
             Gender
Education    Male   Female   Non-Binary   Total
Bachelor's    120      115            5     240
Master's       80       90            3     173
PhD            40       35            2      77
High School    60       65            5     130
Total         300      305           15     620
```

---

### 1️⃣4️⃣ Heatmap of Cross-Tab

Visualizes the **cross-tabulation as a colored grid** — darker = higher count/proportion.

---

### 1️⃣5️⃣ Stacked Bar Chart

Shows the **composition of one categorical variable** within each level of another.

```
Count
│  ░░░███        ░░░░██       ░░░███
│  ░░░███░░░     ░░░░██░░░    ░░░███░░
│  ░░░███░░░███  ░░░░██░░░██  ░░░███░░███
└────────────────────────────────────── Groups
    A              B               C

Legend: ███ = Category 1  ░░░ = Category 2
```

---

### 1️⃣6️⃣ Chi-Square Test of Independence (χ²)

A **statistical test** to determine whether two categorical variables are independent.

**Formula:**
```
χ² = Σ [(Observed - Expected)² / Expected]

Expected frequency = (Row Total × Column Total) / Grand Total
```

**Interpretation:**
```
p-value < 0.05 → Variables are NOT independent (significant association) ✅
p-value ≥ 0.05 → No significant association (variables may be independent)
```

**Assumptions:**
- Each cell expected frequency ≥ 5
- Observations are independent

---

## 📊 Technique Summary Table

| Variable Pair | Technique | Purpose |
|---------------|-----------|---------|
| Num × Num | Scatter Plot | Visualize relationship |
| Num × Num | Pearson / Spearman r | Measure relationship strength |
| Num × Num | Regression Line | Fit linear trend |
| Num × Num | Joint Plot | Scatter + marginal distributions |
| Num × Num | Hexbin | Large dataset density |
| Cat × Num | Box Plot by Group | Distribution comparison |
| Cat × Num | Violin Plot by Group | Shape + summary per group |
| Cat × Num | Bar Chart (mean) | Mean comparison |
| Cat × Num | Strip / Swarm Plot | Show all data points |
| Cat × Cat | Cross-Tabulation | Frequency breakdown |
| Cat × Cat | Heatmap | Visual cross-tab |
| Cat × Cat | Stacked Bar | Composition by group |
| Cat × Cat | Chi-Square Test | Statistical independence |

---

## 🧠 Decision Guide

```
What are the types of your two variables?
    │
    ├── BOTH NUMERICAL
    │       ├── N < 10,000  → Scatter plot + regression line
    │       ├── N ≥ 10,000  → Hexbin plot
    │       ├── Measure strength → Pearson (linear) or Spearman (non-linear/ordinal)
    │       └── Full view → Joint plot
    │
    ├── CATEGORICAL + NUMERICAL
    │       ├── Compare medians   → Box plot by group
    │       ├── Compare shapes    → Violin plot by group
    │       ├── Show all points   → Strip / Swarm plot
    │       └── Compare means     → Bar chart + error bars
    │
    └── BOTH CATEGORICAL
            ├── Count breakdown   → Cross-tabulation
            ├── Visual heatmap    → Heatmap of cross-tab
            ├── Proportions       → Stacked bar chart
            └── Statistical test  → Chi-Square Test
```

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Ignoring non-linearity | Pearson misses curved patterns | Also plot scatter; use Spearman |
| Correlation = causation | Wrong conclusions | Always think about confounders |
| Bar charts hiding spread | Means hide bimodal or skewed data | Add box/violin alongside |
| Overplotting in scatter | Dense cloud unreadable | Use hexbin or alpha transparency |
| Chi-Square with small N | Expected cell < 5 violates assumption | Use Fisher's Exact Test instead |
| Comparing raw counts across groups | Different group sizes mislead | Use proportions / percentages |

---

## 🔗 Related Topics

- `Univariate_Analysis` — Understand each variable alone first
- `Multivariate_Analysis` — Extend to 3+ variables with pair plots and facets
- `Correlation_Analysis` — Deep-dive into full correlation matrices
- `Feature_Engineering` — Use bivariate insights to create interaction features
- `06_Feature_Selection` — Drop features with no relationship to target

---

## 📚 References

- Seaborn Relational Plots: [https://seaborn.pydata.org/tutorial/relational.html](https://seaborn.pydata.org/tutorial/relational.html)
- Seaborn Categorical Plots: [https://seaborn.pydata.org/tutorial/categorical.html](https://seaborn.pydata.org/tutorial/categorical.html)
- SciPy Chi-Square: [https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html)
- Pandas Cross-Tab: [https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html](https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html)
- Pearson vs Spearman: [https://statistics.laerd.com/statistical-guides/pearson-correlation-coefficient-statistical-guide.php](https://statistics.laerd.com/statistical-guides/pearson-correlation-coefficient-statistical-guide.php)
