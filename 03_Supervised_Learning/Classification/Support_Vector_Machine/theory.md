# 📘 Support Vector Machine (SVM) — Theory

---

## 📌 What is SVM?

Support Vector Machine is a **supervised learning algorithm** that finds the  
**optimal hyperplane** (decision boundary) that best separates classes by  
**maximizing the margin** between the closest data points of each class —  
those closest points are called **support vectors**.

```
  Class 0  ●    ●                       Class 1
    ●   ●   ●
       ●         ⊕    ⊕
                   ⊕       ⊕
                       ⊕

         |  ←margin→  |
         |   max this  |
    ─────────────────────── ← Optimal Hyperplane
         |             |

Support vectors: the points closest to the boundary on each side
```

> 💡 "SVM doesn't just find A separating line — it finds THE BEST separating line:  
>     the one with the maximum margin between classes."

---

## 🔍 When to Use SVM?

| Condition | Use SVM? |
|-----------|:--------:|
| High-dimensional data (text, genomics) | ✅ Yes — excellent |
| Small to medium dataset (< 10K rows) | ✅ Yes |
| Non-linear boundaries (with RBF kernel) | ✅ Yes |
| Clear margin of separation expected | ✅ Yes |
| Binary classification | ✅ Yes — native |
| Large dataset (> 100K rows) | ❌ Too slow (O(n²)–O(n³)) |
| Need probability estimates | ⚠️ Requires Platt scaling (slow) |
| Many irrelevant features | ⚠️ Prefer Lasso or tree models |

---

## 🧮 The Hard Margin SVM (Linearly Separable)

For perfectly separable data, SVM maximizes the margin between classes:

```
Decision boundary:   w·x + b = 0
Positive margin:     w·x + b = +1  (Class +1 side)
Negative margin:     w·x + b = −1  (Class −1 side)

Margin width = 2 / ||w||

Objective: Maximize margin = Minimize ||w||²/2

Subject to: yᵢ(w·xᵢ + b) ≥ 1  for all i
              (all points correctly classified and outside the margin)

Where:
  w    = weight vector (normal to the hyperplane)
  b    = bias / intercept
  yᵢ   = class label (+1 or −1)
  xᵢ   = feature vector of sample i
```

---

## 🧮 Soft Margin SVM — The C Parameter

Real data is rarely perfectly separable. **Soft Margin SVM** allows some  
misclassifications by introducing **slack variables (ξᵢ)**:

```
Objective: Minimize  (1/2)||w||² + C × Σξᵢ

Subject to: yᵢ(w·xᵢ + b) ≥ 1 − ξᵢ
             ξᵢ ≥ 0

Where:
  ξᵢ  = slack variable (how far point i is on the wrong side)
  C   = regularization parameter (trade-off between margin and misclassification)

C is large → Penalize misclassifications heavily → Smaller margin, fewer errors → May overfit
C is small → Allow more misclassifications → Larger margin → May underfit
```

### C Parameter Intuition

```
Small C (e.g., C=0.01):
  ─────────────────────────── wide margin
  Allow some misclassification
  More regularization → simpler model

Large C (e.g., C=100):
  ─── narrow margin
  Almost no misclassification allowed
  Less regularization → complex model, may overfit
```

---

## 🌀 The Kernel Trick — Non-Linear SVM

For non-linearly separable data, the **kernel trick** maps data to a  
higher-dimensional space where it becomes linearly separable — **without  
explicitly computing the high-dimensional transformation**:

```
Linear (no kernel):     K(xᵢ, xⱼ) = xᵢ · xⱼ
                        → Linear boundary in original space

Polynomial:             K(xᵢ, xⱼ) = (γ xᵢ · xⱼ + r)^d
                        → Polynomial boundary

RBF (Radial Basis Function / Gaussian):
                        K(xᵢ, xⱼ) = exp(−γ ||xᵢ − xⱼ||²)
                        → Infinite-dimensional mapping
                        → Can separate any distribution given right γ

Sigmoid:                K(xᵢ, xⱼ) = tanh(γ xᵢ · xⱼ + r)
                        → Similar to neural network activation
```

### RBF Kernel — γ Parameter

```
γ controls the "reach" of each training point's influence:

Small γ → each point influences a LARGE area → smooth, simple boundary
Large γ → each point influences a SMALL area → complex, irregular boundary

        Small γ              Large γ
    ___________           ● ●___●___
   /           \         / \_/ \_/ \
  |   Class 1  |         \  _/ \_  /
   \___________/           ‾ ● ‾ ●‾
    (smooth)                (irregular)
```

---

## 📐 Support Vectors — The Key Points

```
Support vectors = the training points that lie on or within the margin
                  (satisfy yᵢ(w·xᵢ + b) ≤ 1)

Properties:
  → Only support vectors determine the decision boundary
  → All other training points are irrelevant once model is fitted
  → Removing a non-support-vector point does NOT change the model
  → Removing a support vector WILL change the model

This makes SVM:
  ✅ Memory efficient — only stores support vectors
  ❌ Slower to predict when many support vectors exist (large C or noisy data)
```

---

## 🌐 Multiclass SVM

SVM is natively binary. Extended to multiclass via:

```
One-vs-Rest (OvR):
  Train K classifiers: class k vs all others
  Predict: class with highest decision function value

One-vs-One (OvO):
  Train K×(K-1)/2 classifiers: every pair of classes
  Predict: majority vote across all classifiers (sklearn default for SVC)
```

---

## 📊 SVM for Regression — SVR

SVM can also perform regression using the **ε-insensitive loss**:

```
Only penalize errors OUTSIDE a tube of width ε:

y
│     ●   ●          ← actual points
│   ●───────────     ← ε-tube
│     ●              ← inside tube: no penalty
│           ●        ← outside tube: penalized
└──────────── x

Objective: Find function f(x) = w·x + b where:
  |yᵢ − f(xᵢ)| ≤ ε   for most points

sklearn: SVR(kernel='rbf', C=1.0, epsilon=0.1)
```

---

## 🎛️ Key Hyperparameters

| Parameter | Effect | Typical Values |
|-----------|--------|----------------|
| `C` | Margin vs misclassification tradeoff | 0.01, 0.1, 1, 10, 100 |
| `kernel` | Feature space transformation | 'linear', 'rbf', 'poly', 'sigmoid' |
| `gamma` | Reach of RBF/poly/sigmoid kernel | 'scale', 'auto', 0.001–10 |
| `degree` | Degree for polynomial kernel | 2, 3, 4 |
| `coef0` | Independent term in poly/sigmoid | 0.0, 1.0 |
| `epsilon` | Width of ε-tube for SVR | 0.01–1.0 |
| `class_weight` | Handle imbalanced classes | 'balanced', None |

---

## 📈 Decision Function

```
SVM outputs a "decision function" value (distance from hyperplane):

  f(x) = w·x + b

  f(x) > 0  →  Class +1  (distance above hyperplane)
  f(x) < 0  →  Class −1  (distance below hyperplane)
  f(x) = 0  →  On the hyperplane (maximum uncertainty)

The larger |f(x)|, the more confident the prediction.

For probability estimates: sklearn uses Platt scaling (5-fold CV internally)
  SVC(probability=True)  →  adds significant training time
```

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Not scaling features | SVM is extremely scale-sensitive | Always StandardScale first |
| Using RBF without tuning γ | Boundary too smooth or too jagged | Use GridSearchCV over C and gamma |
| Large datasets | O(n²)–O(n³) training time | Use LinearSVC or SGDClassifier |
| Default probability=False | Cannot call predict_proba() | Set probability=True (slower) |
| Multiclass without strategy | OvO by default — slow for many classes | Consider LinearSVC with OvR |
| Ignoring class imbalance | Biased toward majority | Use class_weight='balanced' |
| Using sigmoid kernel | Rarely works better than RBF | Default to RBF for non-linear |

---

## 🆚 SVM vs Other Classifiers

| Aspect | SVM | Logistic Reg. | Random Forest | KNN |
|--------|:---:|:-------------:|:-------------:|:---:|
| Training Speed | ❌ Slow (large n) | ✅ Fast | ⚠️ Medium | ✅ Instant |
| Prediction Speed | ✅ Fast | ✅ Fast | ⚠️ Medium | ❌ Slow |
| Feature Scaling | ✅ Required | ✅ Required | ❌ Not needed | ✅ Required |
| Non-linear Boundary | ✅ (kernel) | ❌ | ✅ | ✅ |
| High-Dimensional | ✅ Excellent | ✅ Good | ⚠️ OK | ❌ Poor |
| Small Dataset | ✅ Excellent | ✅ OK | ✅ OK | ✅ Good |
| Interpretability | ❌ Low | ✅ High | ⚠️ Medium | ⚠️ Medium |

---

## 🔗 Related Topics

- `Logistic_Regression` — Also finds a linear boundary, but probabilistic
- `K_Nearest_Neighbors` — Also distance-based classification
- `07_Hyperparameter_Tuning` — GridSearchCV for C and gamma
- `06_Feature_Selection` — SVM with linear kernel gives feature weights

---

## 📚 References

- Scikit-learn `SVC`: [https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- Scikit-learn SVM Guide: [https://scikit-learn.org/stable/modules/svm.html](https://scikit-learn.org/stable/modules/svm.html)
- Original SVM Paper (Cortes & Vapnik, 1995)
- An Introduction to Statistical Learning — Chapter 9
- The Elements of Statistical Learning — Chapter 12
