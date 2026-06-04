# 📘 Logistic Regression — Theory

---

## 📌 What is Logistic Regression?

Despite its name, Logistic Regression is a **classification algorithm** — not a regression model.  
It predicts the **probability** that an observation belongs to a particular class,  
then uses a threshold (default: 0.5) to assign the final class label.

```
Linear Regression:   ŷ = β₀ + β₁x₁ + ... + βₙxₙ          → continuous output
Logistic Regression: P(y=1|X) = σ(β₀ + β₁x₁ + ... + βₙxₙ) → probability [0, 1]
```

> 💡 "Logistic Regression answers: 'What is the probability that this  
>      observation belongs to class 1?' — not 'What is the value?'"

---

## 🔍 When to Use Logistic Regression?

| Condition | Use Logistic Regression? |
|-----------|:-----------------------:|
| Binary classification (0/1, Yes/No) | ✅ Yes — primary use case |
| Multiclass classification | ✅ Yes — via OvR or Softmax |
| Need probability estimates | ✅ Yes — outputs calibrated probabilities |
| Linear decision boundary expected | ✅ Yes |
| Need interpretable coefficients | ✅ Yes |
| Non-linear decision boundary | ❌ No → Use tree models or SVM+kernel |
| Very high-dimensional sparse data | ✅ Yes — works well with L1/L2 |

---

## 🧮 The Model

### Step 1 — Linear Combination (Log-Odds / Logit)

```
z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ  =  Xβ

z is called the log-odds or logit:
  z = log(P(y=1) / P(y=0))  =  log(odds)
```

### Step 2 — Sigmoid Function (Maps z → Probability)

```
         1
σ(z) = ──────     ∈ (0, 1)
       1 + e⁻ᶻ

Properties:
  σ(0)    = 0.5    (decision boundary)
  σ(+∞)  → 1.0
  σ(−∞)  → 0.0
  σ(−z)   = 1 − σ(z)

Sigmoid curve:
  P
  1.0 ─────────────────────────────────────╮
  0.9                                     ╭╯
  0.7                                  ╭──╯
  0.5 ───────────────────────────╭────╯
  0.3                        ╭──╯
  0.1                    ╭───╯
  0.0 ╰───────────────────────────────────
      ────────────────────────────── z
      -6  -4  -2   0   2   4   6
```

### Step 3 — Decision Rule

```
         ┌  Class 1  if  P(y=1|X) ≥ 0.5  (i.e., z ≥ 0)
ŷ =    {
         └  Class 0  if  P(y=1|X) < 0.5  (i.e., z < 0)

Threshold 0.5 is default — can be tuned based on precision/recall tradeoff
```

---

## 🎯 Loss Function — Binary Cross-Entropy (Log Loss)

Logistic Regression minimizes the **log loss** (negative log-likelihood):

```
Loss = − (1/n) × Σ [ yᵢ × log(p̂ᵢ) + (1 − yᵢ) × log(1 − p̂ᵢ) ]

Where:
  yᵢ   = true label (0 or 1)
  p̂ᵢ   = predicted probability P(y=1|xᵢ)

Intuition:
  If y=1 and p̂→1: log(1)=0    → zero loss ✅
  If y=1 and p̂→0: log(0)=−∞  → infinite loss ❌
  If y=0 and p̂→0: log(1)=0   → zero loss ✅
  If y=0 and p̂→1: log(0)=−∞  → infinite loss ❌

→ Model is penalized heavily for confident wrong predictions
```

---

## 🔧 Optimization — Gradient Descent

No closed-form solution for Logistic Regression (unlike Linear Regression).  
Solved iteratively using gradient descent or LBFGS:

```
Gradient of Log Loss w.r.t. β:
  ∂L/∂β = (1/n) × Xᵀ(p̂ − y)

Update rule:
  β ← β − α × (1/n) × Xᵀ(p̂ − y)

Where α = learning rate
```

**sklearn default solver: `lbfgs`** (Limited-memory BFGS) — quasi-Newton method,  
faster than plain gradient descent for small-medium datasets.

---

## 📐 Decision Boundary

Logistic Regression creates a **linear decision boundary** in feature space:

```
Decision boundary: z = β₀ + β₁x₁ + β₂x₂ = 0

Example (2 features):
  x₂ = −(β₀ + β₁x₁) / β₂   ← this is a straight line

  x₂
  │        ● ● ● Class 1 (above boundary)
  │      ●   ●
  │    ●   /── decision boundary
  │       /  
  │      /  ○ ○ ○ Class 0 (below boundary)
  │     / ○  ○
  └──────────────── x₁

With polynomial features → curved decision boundaries
With kernels (non-linear) → complex decision boundaries
```

---

## 🎛️ Regularization

Logistic Regression supports L1 and L2 regularization via the `C` parameter:

```
Loss = Cross-Entropy + (1/C) × Penalty

C = 1/λ   (inverse of regularization strength)

Large C  →  Less regularization  (may overfit)
Small C  →  More regularization  (may underfit)

penalty='l2'  →  Ridge-like  (default, shrinks all coefficients)
penalty='l1'  →  Lasso-like  (zeroes out irrelevant coefficients)
penalty='elasticnet' → Combined L1 + L2
```

---

## 🌐 Multiclass Classification

Logistic Regression extends to multiclass via two strategies:

### 1. One-vs-Rest (OvR)
Trains K binary classifiers — one per class vs all others:
```
Class A vs (B, C, D)
Class B vs (A, C, D)
Class C vs (A, B, D)
...
Final prediction = class with highest probability
```

### 2. Softmax (Multinomial)
Directly models all K classes in a single model:
```
P(y=k|X) = exp(βₖᵀX) / Σⱼ exp(βⱼᵀX)

All probabilities sum to 1: Σₖ P(y=k|X) = 1
```

```python
# sklearn
LogisticRegression(multi_class='ovr')          # One-vs-Rest
LogisticRegression(multi_class='multinomial')   # Softmax
```

---

## 📊 Interpreting Coefficients

### Odds Ratio Interpretation

```
ŷ = σ(β₀ + β₁x₁ + β₂x₂)

Exponentiated coefficient exp(βⱼ) = Odds Ratio

exp(βⱼ) > 1  →  Feature j increases the odds of class 1
exp(βⱼ) < 1  →  Feature j decreases the odds of class 1
exp(βⱼ) = 1  →  Feature j has no effect

Example:
  β₁ = 0.7  →  exp(0.7) = 2.01
  →  Each unit increase in x₁ multiplies the odds of class 1 by 2.01
  →  One unit increase in x₁ doubles the odds of being class 1
```

---

## 📈 Evaluation Metrics for Classification

| Metric | Formula | When to Use |
|--------|---------|------------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Balanced classes |
| **Precision** | TP/(TP+FP) | When FP is costly (spam filter) |
| **Recall** | TP/(TP+FN) | When FN is costly (disease diagnosis) |
| **F1 Score** | 2×P×R/(P+R) | Imbalanced classes |
| **ROC-AUC** | Area under ROC | Threshold-independent performance |
| **Log Loss** | −mean(y×log(ŷ)) | Probabilistic evaluation |

### Confusion Matrix
```
                 Predicted
                 0        1
Actual  0  [ TN=850  FP=50  ]   ← predicted positive but actually negative
        1  [ FN=30   TP=70  ]   ← predicted negative but actually positive

TN = True Negative   TP = True Positive
FP = False Positive  FN = False Negative

Precision = TP/(TP+FP) = 70/120 = 0.583
Recall    = TP/(TP+FN) = 70/100 = 0.700
Accuracy  = (TN+TP)/Total = 920/1000 = 0.920
```

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Not scaling features | Slow convergence, poor results | Always StandardScale before LR |
| Imbalanced classes | Predicts majority class | Use class_weight='balanced' or SMOTE |
| Collinear features | Unstable coefficients | Use L2 regularization or drop features |
| Non-linear boundary | Poor fit despite tuning | Add polynomial features or use tree model |
| Using raw coefficients as importance | Scale-dependent | Use standardized coefficients |
| Default threshold of 0.5 | Sub-optimal P/R tradeoff | Tune threshold using precision-recall curve |
| Convergence warning | Model didn't fully converge | Increase max_iter or scale features |

---

## 🆚 Logistic Regression vs Other Classifiers

| Aspect | Logistic Reg. | Decision Tree | Random Forest | XGBoost |
|--------|:------------:|:------------:|:-------------:|:-------:|
| Interpretability | ✅ High | ✅ High | ⚠️ Medium | ❌ Low |
| Training Speed | ✅ Fast | ✅ Fast | ⚠️ Medium | ✅ Fast |
| Nonlinear Boundary | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| Probability Output | ✅ Yes | ⚠️ Rough | ⚠️ Rough | ✅ Yes |
| Feature Scaling Needed | ✅ Yes | ❌ No | ❌ No | ❌ No |
| Handles Imbalance | ⚠️ With help | ⚠️ With help | ⚠️ With help | ⚠️ With help |

---

## 🔗 Related Topics

- `Linear_Regression` — Logistic Regression shares the linear combination step
- `K_Nearest_Neighbors` — Non-parametric alternative
- `Decision_Trees` — Non-linear classification without feature scaling
- `Support_Vector_Machine` — Also finds a linear decision boundary (with kernels for non-linear)
- `05_Model_Evaluation` — ROC-AUC, confusion matrix, classification report
- `06_Feature_Selection` — L1 Logistic Regression for embedded feature selection

---

## 📚 References

- Scikit-learn `LogisticRegression`: [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Scikit-learn Classification Metrics: [https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- An Introduction to Statistical Learning — Chapter 4
- The Elements of Statistical Learning — Chapter 4
