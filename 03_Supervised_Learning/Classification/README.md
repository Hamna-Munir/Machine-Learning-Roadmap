# 🎯 Classification

## Overview

Classification is a **supervised learning task** where the goal is to predict a  
**discrete class label** for each input — assigning observations into predefined categories.

```
Input Features (X)  →  Classification Model  →  Predicted Class Label (ŷ)

Examples:
  Email features     →  Spam / Not Spam         (Binary)
  Medical symptoms   →  Disease A / B / C       (Multiclass)
  Image pixels       →  Cat / Dog / Bird / ...  (Multiclass)
  Transaction data   →  Fraud / Legitimate      (Binary, Imbalanced)
```

This module covers **10 classification algorithms** — from simple probabilistic  
models to powerful gradient-boosted ensembles used in industry and competitions.

---

## 📁 Folder Structure

```
Classification/
│
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

Each subfolder contains:

- 📘 `theory.md` → Algorithm explanation, formula, and assumptions
- 📓 `.ipynb` → Interactive experiments, visualizations & evaluation
- 🐍 `.py` → Reusable, modular Python implementation

---

## 📌 Algorithms Covered

---

### 1. Logistic Regression

A **linear classifier** that models the **probability** of class membership  
using the sigmoid function. Despite its name, it is a classification algorithm.

- **Core:** `P(y=1|X) = σ(β₀ + β₁x₁ + ... + βₙxₙ)`
- **Decision boundary:** Linear hyperplane
- **Loss:** Binary cross-entropy (log loss)
- **Multiclass:** One-vs-Rest (OvR) or Softmax
- **Best for:** Linearly separable data, interpretable baseline model

---

### 2. K-Nearest Neighbors (KNN)

A **non-parametric, lazy learning** algorithm — classifies new points based on  
the majority class among their K closest neighbors in feature space.

- **Core:** Distance-based — Euclidean, Manhattan, Minkowski
- **No training phase:** Stores the entire training set
- **Decision boundary:** Non-linear, highly flexible
- **Key hyperparameter:** K (number of neighbors)
- **Best for:** Small datasets, non-linear boundaries, quick prototyping

---

### 3. Naive Bayes

A **probabilistic classifier** based on Bayes' theorem — assumes all features  
are **conditionally independent** given the class label (the "naive" assumption).

- **Core:** `P(y|X) ∝ P(y) × ΠP(xᵢ|y)`
- **Variants:** Gaussian (continuous), Multinomial (counts), Bernoulli (binary)
- **Training:** Extremely fast — only computes class statistics
- **Best for:** Text classification, spam filtering, high-dimensional data

---

### 4. Decision Trees

A **tree-based model** that recursively splits data based on feature thresholds —  
creating a sequence of if/else decision rules that are highly interpretable.

- **Splitting criteria:** Gini Impurity, Entropy (Information Gain)
- **Overfitting risk:** High — requires pruning or depth limits
- **Decision boundary:** Axis-aligned step functions
- **Best for:** Interpretability, categorical features, non-linear patterns

---

### 5. Random Forest

An **ensemble of decision trees** trained on random subsets of data and features  
using **bagging** — reduces variance and overfitting dramatically.

- **Core:** Bootstrap Aggregating (Bagging) + Random Feature Subsampling
- **Feature importance:** Mean Decrease in Impurity (MDI)
- **Handles:** Missing values, non-linear patterns, high dimensionality
- **Best for:** General-purpose tabular data, robust baseline ensemble

---

### 6. Support Vector Machine (SVM)

Finds the **maximum-margin hyperplane** that best separates classes —  
powerful for high-dimensional and small-sample classification tasks.

- **Core:** Maximize margin between support vectors of each class
- **Kernel trick:** RBF, Polynomial, Linear, Sigmoid — handles non-linearity
- **Soft margin:** C parameter controls margin vs misclassification tradeoff
- **Best for:** High-dimensional data, small datasets, binary classification

---

### 7. Gradient Boosting

Builds an ensemble of **weak learners (shallow trees) sequentially** — each tree  
corrects the residual errors of the previous one using gradient descent.

- **Core:** Functional gradient descent — minimize loss in function space
- **Key params:** n_estimators, learning_rate, max_depth
- **Risk:** Overfitting at high n_estimators without regularization
- **Best for:** Tabular data, when accuracy is the primary goal

---

### 8. XGBoost

An **optimized, regularized gradient boosting** framework — faster, more scalable,  
and typically more accurate than sklearn's GradientBoosting.

- **Improvements over GBM:** L1+L2 regularization, second-order gradients, column subsampling
- **Handles:** Missing values natively, sparse data
- **Speed:** Parallel tree construction, cache-aware computation
- **Best for:** Kaggle competitions, structured/tabular data, large datasets

---

### 9. LightGBM

A **histogram-based gradient boosting** framework by Microsoft —  
uses leaf-wise (best-first) tree growth for faster training on large datasets.

- **Key innovations:** GOSS (Gradient-based One-Side Sampling), EFB (Exclusive Feature Bundling)
- **Growth:** Leaf-wise (deeper, more complex) vs level-wise (more balanced)
- **Speed:** Significantly faster than XGBoost on large datasets
- **Best for:** Large datasets, high-cardinality categorical features, speed-critical applications

---

### 10. CatBoost

A gradient boosting library by Yandex with **native categorical feature support** —  
no manual label encoding or one-hot encoding required.

- **Key innovation:** Ordered boosting — prevents target leakage during training
- **Trees:** Symmetric (oblivious) trees — faster prediction
- **Categorical:** Built-in target statistics encoding
- **Best for:** Datasets with many categorical features, minimal preprocessing needed

---

## 📊 Algorithm Comparison

| Algorithm | Training Speed | Prediction Speed | Handles Nonlinearity | Interpretable | Feature Scaling | Best For |
|-----------|:--------------:|:----------------:|:--------------------:|:-------------:|:---------------:|----------|
| Logistic Regression | ✅ Fast | ✅ Fast | ❌ No | ✅ High | ✅ Required | Baseline, interpretability |
| KNN | ✅ Instant | ❌ Slow | ✅ Yes | ⚠️ Moderate | ✅ Required | Small datasets |
| Naive Bayes | ✅ Very Fast | ✅ Very Fast | ❌ No | ✅ High | ❌ Not needed | Text, high-dim |
| Decision Tree | ✅ Fast | ✅ Fast | ✅ Yes | ✅ High | ❌ Not needed | Interpretability |
| Random Forest | ⚠️ Medium | ⚠️ Medium | ✅ Yes | ⚠️ Moderate | ❌ Not needed | General purpose |
| SVM | ❌ Slow (large n) | ⚠️ Medium | ✅ (kernel) | ❌ Low | ✅ Required | High-dim, small n |
| Gradient Boosting | ⚠️ Medium | ✅ Fast | ✅ Yes | ❌ Low | ❌ Not needed | Tabular accuracy |
| XGBoost | ✅ Fast | ✅ Fast | ✅ Yes | ❌ Low | ❌ Not needed | Competitions |
| LightGBM | ✅ Very Fast | ✅ Very Fast | ✅ Yes | ❌ Low | ❌ Not needed | Large datasets |
| CatBoost | ✅ Fast | ✅ Fast | ✅ Yes | ❌ Low | ❌ Not needed | Categorical-heavy |

---

## 📏 Classification Evaluation Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| **Accuracy** | (TP+TN) / Total | Misleading on imbalanced datasets |
| **Precision** | TP / (TP+FP) | Penalizes false positives |
| **Recall (Sensitivity)** | TP / (TP+FN) | Penalizes false negatives |
| **F1 Score** | 2×(P×R)/(P+R) | Harmonic mean of Precision & Recall |
| **ROC-AUC** | Area under ROC | Threshold-independent performance |
| **Log Loss** | −mean(y×log(ŷ)) | Penalizes confident wrong predictions |
| **MCC** | (TP×TN−FP×FN)/√... | Best single metric for imbalanced classes |
| **Confusion Matrix** | TP / FP / FN / TN | Full breakdown of classification errors |

### Choosing the Right Metric

```
Balanced classes          →  Accuracy, F1 Macro
Imbalanced classes        →  F1 (weighted), ROC-AUC, MCC
High cost of false pos.   →  Precision (e.g., spam detection)
High cost of false neg.   →  Recall    (e.g., cancer diagnosis)
Probabilistic output      →  Log Loss, ROC-AUC
```

---

## 🧠 Decision Guide: Which Algorithm to Use?

```
Dataset size?
    │
    ├── Small (< 1,000)
    │       ├── Need interpretability    → Logistic Regression / Decision Tree
    │       ├── Non-linear boundary      → KNN / SVM (RBF kernel)
    │       └── Probabilistic output     → Naive Bayes
    │
    ├── Medium (1K–100K)
    │       ├── Interpretability needed  → Decision Tree / Logistic Regression
    │       ├── Best accuracy            → Random Forest / XGBoost
    │       └── Many categorical feats   → CatBoost
    │
    └── Large (> 100K)
            ├── Speed priority           → LightGBM
            ├── Best accuracy            → XGBoost / LightGBM
            ├── Categorical heavy        → CatBoost / LightGBM
            └── Text / sparse data       → Naive Bayes / Logistic Regression
```

---

## 🎯 Learning Objectives

By completing this module, you will understand:

✔ The mathematical foundation of each classification algorithm  
✔ How each model learns a **decision boundary** from training data  
✔ When to use each algorithm based on data size, type, and requirements  
✔ How to evaluate classifiers using appropriate metrics  
✔ How to handle **class imbalance** in classification tasks  
✔ Why **ensemble methods** (Random Forest, Boosting) outperform single models  
✔ How to implement, tune, and deploy each algorithm using scikit-learn  

---

## 🛠 Tools & Libraries

- Python 3.x
- Pandas, NumPy
- Scikit-learn
- XGBoost (`xgboost`)
- LightGBM (`lightgbm`)
- CatBoost (`catboost`)
- Matplotlib, Seaborn
- Imbalanced-learn (`imblearn`) — for class imbalance handling

---

## 📌 Usage

Each folder contains:

- `theory.md` → Algorithm concept, formula, assumptions, pros & cons
- `.ipynb` → Full experiment: data, training, evaluation, visualization
- `.py` → Clean, reusable implementation for production pipelines

### Recommended Workflow:
1. Read `theory.md` — understand the algorithm's core idea
2. Run `.ipynb` — experiment, visualize decision boundaries, evaluate metrics
3. Use `.py` — integrate into full ML pipelines with cross-validation

---

## 🚀 Importance in Machine Learning

Classification is the **most common ML task** in industry:

- **Fraud Detection** → binary, highly imbalanced
- **Medical Diagnosis** → multiclass, high recall priority
- **Spam Filtering** → binary, Naive Bayes baseline
- **Customer Churn** → binary, interpretability needed
- **Image Recognition** → multiclass, deep learning + boosting
- **Credit Scoring** → binary, regulatory interpretability required
- **Sentiment Analysis** → multiclass, text features

---

## 📈 Recommended Learning Order

```
Logistic Regression  →  KNN  →  Naive Bayes  →  Decision Tree
         ↓
Random Forest  →  SVM  →  Gradient Boosting
         ↓
XGBoost  →  LightGBM  →  CatBoost
```

---

## 📈 Next Steps

After completing this section, move to:

- `05_Model_Evaluation` → Confusion matrix, ROC-AUC, cross-validation
- `06_Feature_Selection` → Select features to improve classifier performance
- `07_Hyperparameter_Tuning` → GridSearchCV, RandomSearchCV, Bayesian Optimization
- `08_Ensemble_Learning` → Stacking, Voting, and advanced ensembles

---

## 🤝 Contribution

This repository is part of a structured learning journey.  
Suggestions for improvements are always welcome.

---

## ⭐ Support

If you find this helpful, consider giving the repository a ⭐ on GitHub.
