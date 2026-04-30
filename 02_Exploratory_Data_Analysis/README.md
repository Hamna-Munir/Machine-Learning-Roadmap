# 📊 Exploratory Data Analysis (EDA)

## Overview

Exploratory Data Analysis is the **second and equally critical step** in any Machine Learning pipeline.  
It involves **understanding, summarizing, and visualizing** the structure and patterns within data  
before applying any machine learning model.

This module focuses on essential EDA techniques used in real-world ML workflows.

---

## 📁 Folder Structure

This section contains structured implementations of key EDA techniques:

- Univariate Analysis
- Bivariate Analysis
- Multivariate Analysis
- Correlation Analysis

Each topic includes:

- 📘 `theory.md` → Concept explanation
- 📓 `.ipynb` → Interactive experiments & visualization

---

## 📌 Topics Covered

### 1. Univariate Analysis

Analyzing **one variable at a time** to understand its distribution and properties:

- Histograms & Frequency Distributions
- Box Plots & Violin Plots
- Measures of Central Tendency (Mean, Median, Mode)
- Measures of Spread (Variance, Std Dev, IQR)
- Skewness & Kurtosis
- KDE (Kernel Density Estimation) Plots

---

### 2. Bivariate Analysis

Exploring the **relationship between two variables**:

- Scatter Plots (numerical vs numerical)
- Bar Charts & Grouped Bar Charts (categorical vs numerical)
- Box Plots by Category (categorical vs numerical)
- Cross-Tabulation & Heatmaps (categorical vs categorical)
- Line Plots (trend over time)

---

### 3. Multivariate Analysis

Understanding **interactions among three or more variables**:

- Pair Plots (Seaborn `pairplot`)
- Hue-colored Scatter Plots
- Facet Grids
- 3D Scatter Plots
- Parallel Coordinates
- Bubble Charts

---

### 4. Correlation Analysis

Measuring the **strength and direction** of relationships between variables:

- Pearson Correlation (linear, continuous)
- Spearman Correlation (monotonic, ordinal)
- Kendall's Tau (non-parametric)
- Correlation Heatmaps
- Point-Biserial Correlation (binary vs continuous)
- VIF (Variance Inflation Factor) for Multicollinearity

---

## 🎯 Learning Objectives

By completing this module, you will understand:

✔ How to describe and summarize individual features  
✔ How to detect patterns, trends, and relationships in data  
✔ How to identify outliers and anomalies visually  
✔ How to assess feature-target relationships before modeling  
✔ How to detect multicollinearity between features  
✔ Practical implementation of EDA using Python visualization libraries  

---

## 🛠 Tools & Libraries

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly *(optional — for interactive charts)*
- Scipy *(for statistical tests)*
- Scikit-learn *(for VIF and preprocessing)*

---

## 📌 Usage

Each folder contains:

- `theory.md` → Concept explanation
- `.ipynb` → Hands-on experiments & visualizations

Example workflow:

1. Read theory
2. Run notebook experiments
3. Apply EDA insights to inform preprocessing decisions

---

## 🔄 EDA Workflow

```
Load Data
    │
    ▼
Univariate Analysis       → Understand each feature individually
    │
    ▼
Bivariate Analysis        → Explore pairwise relationships
    │
    ▼
Multivariate Analysis     → Detect complex interactions
    │
    ▼
Correlation Analysis      → Measure feature-target strength
    │
    ▼
Document Findings         → Drive feature engineering decisions
    │
    ▼
Proceed to Preprocessing / Modeling
```

---

## 🚀 Importance in Machine Learning

Exploratory Data Analysis is crucial because:

- It reveals the **shape, distribution, and quality** of your data
- It uncovers **hidden patterns** that guide feature engineering
- It helps detect **outliers, missing values, and anomalies** early
- It identifies **correlated features** that may cause multicollinearity
- It informs **model selection** by revealing linearity or non-linearity
- Skipping EDA leads to **poor model performance** and unexpected results

---

## 📈 Next Steps

After completing this section, move to:

- `03_Supervised_Learning` → Regression & Classification algorithms
- `06_Feature_Selection` → Select best features informed by EDA insights
- `05_Model_Evaluation` → Evaluate models built on well-understood data

---

## 🤝 Contribution

This repository is part of a structured learning journey.  
Suggestions for improvements are always welcome.

---

## ⭐ Support

If you find this helpful, consider giving the repository a ⭐ on GitHub.
