# 🤖 Supervised Learning

## Overview

Supervised Learning is the **core of predictive machine learning** — the process of training  
a model on **labeled data** (input-output pairs) so it can learn the mapping from features  
to a target variable and make accurate predictions on unseen data.

```
Training Data (X, y)  →  Model Training  →  Learned Function f(X)  →  Predictions ŷ
```

This module covers the two fundamental supervised learning tasks — **Regression** and **Classification** —  
with implementations for the most widely used algorithms in industry and research.

---

## 📁 Folder Structure

```
03_Supervised_Learning/
│
├── Regression/
│   ├── Linear_Regression/
│   ├── Polynomial_Regression/
│   ├── Ridge_Regression/
│   ├── Lasso_Regression/
│   └── ElasticNet/
│
└── Classification/
    ├── Logistic_Regression/
    ├── K_Nearest_Neighbors/
    ├── Naive_Bayes/
    ├── Decision_Trees/
    ├── Random_Forest/
    ├── Support_Vector_Machine/
    ├── Gradient_Boosting/
    ├── XGBoost/
    ├── LightGBM/
    └── CatBoost/
```

Each topic includes:

- 📘 `theory.md` → Concept explanation, formula, assumptions, pros & cons
- 📓 `.ipynb` → Interactive experiments, visualizations & evaluation
- 🐍 `.py` → Reusable, modular Python implementation

---

## 📌 Topics Covered

---

### 📈 PART A — Regression

Regression algorithms predict a **continuous numerical output**.

---

#### 1. Linear Regression
The foundation of supervised learning — models the relationship between features  
and a continuous target as a straight line (or hyperplane).
- Ordinary Least Squares (OLS)
- Gradient Descent optimization
- Assumptions: linearity, homoscedasticity, no multicollinearity
- Metrics: MAE, MSE, RMSE, R²

---

#### 2. Polynomial Regression
Extends Linear Regression by adding **polynomial feature terms** —  
captures non-linear relationships while remaining a linear model in parameters.
- Degree selection and overfitting
- Bias-variance tradeoff
- Pipeline with PolynomialFeatures + LinearRegression

---

#### 3. Ridge Regression (L2 Regularization)
Adds an **L2 penalty** (sum of squared coefficients) to the loss function —  
shrinks coefficients to prevent overfitting and handle multicollinearity.
- L2 penalty term: λ × Σβ²
- Cross-validation for optimal λ (alpha)
- Keeps all features but shrinks coefficients toward zero

---

#### 4. Lasso Regression (L1 Regularization)
Adds an **L1 penalty** (sum of absolute coefficients) — performs automatic  
**feature selection** by shrinking some coefficients to exactly zero.
- L1 penalty term: λ × Σ|β|
- Sparse solutions — built-in feature selection
- Coordinate descent optimization

---

#### 5. ElasticNet
Combines **L1 + L2 regularization** — balances feature selection (Lasso)  
with coefficient stability (Ridge).
- Mixed penalty: α × L1 + (1−α) × L2
- l1_ratio parameter controls the mix
- Best of both Ridge and Lasso

---

### 🎯 PART B — Classification

Classification algorithms predict a **discrete class label** or probability.

---

#### 6. Logistic Regression
Despite its name, a **binary classification** algorithm — models the probability  
of class membership using the sigmoid function.
- Sigmoid / softmax function
- Log-loss (binary cross-entropy) optimization
- Multiclass via OvR or softmax

---

#### 7. K-Nearest Neighbors (KNN)
A **non-parametric, instance-based** algorithm — classifies a new point based  
on the majority class of its K nearest neighbors.
- Distance metrics: Euclidean, Manhattan, Minkowski
- No training phase — lazy learner
- Sensitive to feature scaling and K selection

---

#### 8. Naive Bayes
A **probabilistic classifier** based on Bayes' theorem — assumes all features  
are conditionally independent given the class label.
- Gaussian, Multinomial, Bernoulli variants
- Very fast training and prediction
- Performs well on text classification

---

#### 9. Decision Trees
A **tree-based** model that recursively splits data based on feature thresholds —  
highly interpretable but prone to overfitting.
- Splitting criteria: Gini Impurity, Entropy (Information Gain)
- Pre-pruning (max_depth) and post-pruning
- Handles both numerical and categorical features

---

#### 10. Random Forest
An **ensemble of decision trees** trained on random subsets of data and features —  
reduces variance through bagging and random feature selection.
- Bootstrap aggregation (Bagging)
- Feature importance via mean decrease impurity
- Robust to overfitting — one of the best general-purpose algorithms

---

#### 11. Support Vector Machine (SVM)
Finds the **maximum-margin hyperplane** that best separates classes —  
powerful for high-dimensional and small-sample datasets.
- Hard and soft margin SVM
- Kernel trick: Linear, RBF, Polynomial, Sigmoid
- C parameter controls margin vs misclassification tradeoff

---

#### 12. Gradient Boosting
Builds an ensemble of trees **sequentially** — each tree corrects the errors  
of the previous one using gradient descent in function space.
- Boosting: weak learners → strong learner
- Learning rate, n_estimators, max_depth tuning
- sklearn GradientBoostingClassifier / Regressor

---

#### 13. XGBoost
An **optimized, regularized** implementation of gradient boosting —  
faster, more scalable, and often higher accuracy than vanilla gradient boosting.
- Second-order gradient optimization
- L1 + L2 regularization built-in
- Handles missing values natively
- Column subsampling for variance reduction

---

#### 14. LightGBM
A **histogram-based** gradient boosting framework by Microsoft —  
leaf-wise tree growth for faster training on large datasets.
- Leaf-wise (best-first) tree growth vs level-wise
- GOSS (Gradient-based One-Side Sampling)
- EFB (Exclusive Feature Bundling)
- Extremely fast on large, high-dimensional datasets

---

#### 15. CatBoost
A gradient boosting library by Yandex with **native categorical feature support** —  
no manual encoding required.
- Ordered boosting to prevent target leakage
- Symmetric (oblivious) trees
- Built-in categorical encoding
- Minimal hyperparameter tuning required

---

## 🎯 Learning Objectives

By completing this module, you will understand:

✔ The difference between **Regression** and **Classification** tasks  
✔ How each algorithm learns from data — formula, objective, and optimization  
✔ When to use each algorithm based on data size, dimensionality, and problem type  
✔ How to evaluate models using appropriate metrics  
✔ How **regularization** prevents overfitting (Ridge, Lasso, ElasticNet)  
✔ Why **ensemble methods** outperform single models (Random Forest, Boosting)  
✔ How to implement, tune, and deploy each algorithm using scikit-learn  

---

## 📊 Algorithm Comparison at a Glance

### Regression

| Algorithm | Handles Nonlinearity | Regularization | Interpretable | Best For |
|-----------|:-------------------:|:--------------:|:-------------:|----------|
| Linear Regression | ❌ | ❌ | ✅ | Baseline, linear data |
| Polynomial Regression | ✅ | ❌ | ✅ | Curved relationships |
| Ridge | ❌ | L2 ✅ | ✅ | Multicollinearity |
| Lasso | ❌ | L1 ✅ | ✅ | Feature selection |
| ElasticNet | ❌ | L1+L2 ✅ | ✅ | Many correlated features |

### Classification

| Algorithm | Training Speed | Interpretable | Handles Nonlinearity | Best For |
|-----------|:--------------:|:-------------:|:--------------------:|----------|
| Logistic Regression | ✅ Fast | ✅ | ❌ | Baseline, linear boundary |
| KNN | ✅ Instant | ⚠️ | ✅ | Small datasets |
| Naive Bayes | ✅ Very Fast | ✅ | ❌ | Text, high-dimensional |
| Decision Tree | ✅ Fast | ✅ | ✅ | Interpretability needed |
| Random Forest | ⚠️ Medium | ⚠️ | ✅ | General purpose |
| SVM | ⚠️ Slow on large | ❌ | ✅ (kernel) | High-D, small datasets |
| Gradient Boosting | ⚠️ Medium | ❌ | ✅ | Tabular data |
| XGBoost | ✅ Fast | ❌ | ✅ | Competition, tabular |
| LightGBM | ✅ Very Fast | ❌ | ✅ | Large datasets |
| CatBoost | ✅ Fast | ❌ | ✅ | Categorical-heavy data |

---

## 📏 Evaluation Metrics

### Regression Metrics
| Metric | Formula | Notes |
|--------|---------|-------|
| **MAE** | mean\|yᵢ − ŷᵢ\| | Robust to outliers |
| **MSE** | mean(yᵢ − ŷᵢ)² | Penalizes large errors |
| **RMSE** | √MSE | Same units as target |
| **R²** | 1 − SS_res/SS_tot | 1.0 = perfect fit |
| **Adj. R²** | Penalizes extra features | Fairer for multiple features |

### Classification Metrics
| Metric | Notes |
|--------|-------|
| **Accuracy** | Correct predictions / total — misleading if imbalanced |
| **Precision** | TP / (TP + FP) — penalizes false positives |
| **Recall** | TP / (TP + FN) — penalizes false negatives |
| **F1 Score** | Harmonic mean of Precision & Recall |
| **ROC-AUC** | Area under ROC curve — threshold-independent |
| **Log Loss** | Probabilistic loss — penalizes confident wrong predictions |

---

## 🛠 Tools & Libraries

- Python 3.x
- Pandas, NumPy
- Scikit-learn
- XGBoost (`xgboost`)
- LightGBM (`lightgbm`)
- CatBoost (`catboost`)
- Matplotlib, Seaborn
- Scipy

---

## 📌 Usage

Each folder contains:

- `theory.md` → Algorithm explanation, formula, and assumptions
- `.ipynb` → Full experiment with data, training, evaluation, and visualization
- `.py` → Clean, reusable implementation for production pipelines

### Recommended Workflow:
1. Read `theory.md` — understand the algorithm
2. Run `.ipynb` — experiment and visualize
3. Use `.py` — integrate into your ML pipeline

---

## 🚀 Importance in Machine Learning

Supervised Learning algorithms are crucial because:

- They solve the most common real-world ML problems — prediction and classification
- Understanding each algorithm's **strengths, weaknesses, and assumptions** drives better model selection
- Knowing **when regularization is needed** prevents costly overfitting mistakes
- **Ensemble methods** (Random Forest, XGBoost, LightGBM) dominate structured/tabular data competitions
- **Interpretable models** (Linear, Logistic, Decision Tree) are required in regulated industries

---

## 📈 Recommended Learning Order

```
Linear Regression  →  Polynomial Regression  →  Ridge  →  Lasso  →  ElasticNet
        ↓
Logistic Regression  →  KNN  →  Naive Bayes  →  Decision Tree
        ↓
Random Forest  →  SVM  →  Gradient Boosting
        ↓
XGBoost  →  LightGBM  →  CatBoost
```

---

## 📈 Next Steps

After completing this section, move to:

- `04_Unsupervised_Learning` → Clustering and dimensionality reduction
- `05_Model_Evaluation` → Cross-validation, ROC-AUC, confusion matrix
- `06_Feature_Selection` → Select features to improve model performance
- `07_Hyperparameter_Tuning` → GridSearchCV, RandomSearchCV, Bayesian Optimization
- `08_Ensemble_Learning` → Stacking, bagging, and advanced boosting

---

## 🤝 Contribution

This repository is part of a structured learning journey.  
Suggestions for improvements are always welcome.

---

## ⭐ Support

If you find this helpful, consider giving the repository a ⭐ on GitHub.
