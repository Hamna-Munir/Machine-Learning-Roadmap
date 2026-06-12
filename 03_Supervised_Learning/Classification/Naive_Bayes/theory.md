# 📘 Naive Bayes — Theory

---

## 📌 What is Naive Bayes?

Naive Bayes is a **probabilistic classification algorithm** based on **Bayes' Theorem**.  
It predicts the class with the **highest posterior probability** given the input features,  
making the "naive" assumption that all features are **conditionally independent** given the class.

```
Bayes' Theorem:
                P(X | y) × P(y)
P(y | X)  =  ─────────────────
                    P(X)

Where:
  P(y | X)  = Posterior  → probability of class y given features X
  P(X | y)  = Likelihood → probability of observing X given class y
  P(y)      = Prior      → probability of class y (from training data)
  P(X)      = Evidence   → constant (same for all classes, can ignore)
```

**Classification Rule:**
```
ŷ = argmax_y  P(y) × Π P(xᵢ | y)
                        i=1..n

→ Pick the class that maximizes the product of:
  (class prior) × (likelihood of each feature given that class)
```

> 💡 "Naive Bayes is naive because real features are rarely independent —  
>      yet despite this bold assumption, it works surprisingly well in practice."

---

## 🔍 When to Use Naive Bayes?

| Condition | Use Naive Bayes? |
|-----------|:---------------:|
| Text classification (spam, sentiment) | ✅ Yes — primary use case |
| High-dimensional sparse data | ✅ Yes — scales very well |
| Small training dataset | ✅ Yes — few parameters to estimate |
| Real-time prediction needed | ✅ Yes — extremely fast |
| Features are truly independent | ✅ Yes |
| Features are highly correlated | ⚠️ Caution — assumption violated |
| Continuous features with non-Gaussian distribution | ❌ → Transform or use GNB carefully |
| Complex non-linear decision boundary | ❌ → Use tree models |

---

## 🧮 The Naive Independence Assumption

```
Full joint probability:
  P(x₁, x₂, ..., xₙ | y) is intractable for large n

Naive assumption:
  P(x₁, x₂, ..., xₙ | y) ≈ Π P(xᵢ | y)
                             i=1..n

  → Each feature is independent of all others given the class label

Why it works despite being wrong:
  - Even if correlations exist, the ranking of P(y|X) across classes
    is often preserved (correct class still gets highest probability)
  - Works especially well when features carry independent signals
```

---

## 📊 Three Variants of Naive Bayes

---

### 1️⃣ Gaussian Naive Bayes (GaussianNB)

**Use for:** Continuous numerical features  
**Assumption:** Each feature follows a **Gaussian (normal) distribution** within each class

```
P(xᵢ | y) = (1 / √(2π σ²_iy)) × exp(−(xᵢ − μ_iy)² / 2σ²_iy)

Where:
  μ_iy  = mean of feature i for class y
  σ²_iy = variance of feature i for class y

Training: Estimate μ and σ² for each feature per class
```

**Example:**
```
Feature: Age
  Class 0 (No Churn): μ=35, σ=8
  Class 1 (Churn):    μ=28, σ=6

For a new person with Age=30:
  P(Age=30 | Class 0) = Gaussian(30; μ=35, σ=8) = 0.043
  P(Age=30 | Class 1) = Gaussian(30; μ=28, σ=6) = 0.054
  → Age=30 is more likely under Class 1
```

---

### 2️⃣ Multinomial Naive Bayes (MultinomialNB)

**Use for:** Count data — word counts, term frequencies  
**Assumption:** Features represent **integer counts** (e.g., word occurrence counts)

```
P(xᵢ | y) = (count(xᵢ, y) + α) / (count(y) + α × n_features)

Where:
  count(xᵢ, y) = total count of feature i in class y documents
  α             = Laplace smoothing parameter (default: 1)
  n_features    = vocabulary size

Laplace smoothing (α > 0):
  Prevents P(xᵢ|y) = 0 for words not seen in training
  α = 0: no smoothing
  α = 1: Laplace smoothing (add-one)
  α → ∞: uniform distribution
```

**Typical pipeline:**
```python
TfidfVectorizer / CountVectorizer → MultinomialNB
```

---

### 3️⃣ Bernoulli Naive Bayes (BernoulliNB)

**Use for:** Binary features — word presence/absence (0 or 1)  
**Assumption:** Features are **binary** — does the feature occur or not?

```
P(xᵢ | y) = P(xᵢ=1 | y)^xᵢ × (1 − P(xᵢ=1 | y))^(1−xᵢ)

Where:
  P(xᵢ=1 | y) = (count(xᵢ=1, y) + α) / (count(y) + 2α)

Key difference from Multinomial:
  Bernoulli explicitly models ABSENCE of features
  Multinomial ignores features not present in the document
```

**When to choose Bernoulli over Multinomial:**
```
Short documents → BernoulliNB often better
Long documents  → MultinomialNB often better
Presence/absence signals → BernoulliNB
Frequency signals        → MultinomialNB
```

---

### 4️⃣ Complement Naive Bayes (ComplementNB)

A variant of Multinomial NB designed for **imbalanced class distributions**:

```
Instead of P(xᵢ | y), uses P(xᵢ | NOT y)

→ Estimates parameters from ALL other classes
→ More robust when one class dominates the training set
```

---

## 📐 Decision Boundary

Despite the naive assumption, Naive Bayes can produce **non-linear decision boundaries**  
(especially Gaussian NB with different variances per class):

```
GaussianNB with equal variances:  linear boundary (like Logistic Regression)
GaussianNB with unequal variances: quadratic boundary
MultinomialNB / BernoulliNB:      linear boundary in log-probability space
```

---

## 🔢 Log-Probability for Numerical Stability

In practice, probabilities are multiplied across many features, which can cause **underflow**:

```
P(y) × P(x₁|y) × P(x₂|y) × ... × P(xₙ|y)  → very small number!

Solution: Work in log space
  log P(y|X) ∝ log P(y) + Σ log P(xᵢ|y)
  → Sum instead of product → no underflow ✅
  → Prediction: argmax of log-probabilities (same result)
```

---

## 🎛️ Key Hyperparameters

| Variant | Key Parameter | Description |
|---------|:------------:|-------------|
| **GaussianNB** | `var_smoothing` | Adds to variance — prevents division by zero |
| **MultinomialNB** | `alpha` | Laplace smoothing — prevents zero probabilities |
| **BernoulliNB** | `alpha` | Laplace smoothing |
| **BernoulliNB** | `binarize` | Threshold to binarize continuous features |
| **All** | `priors` | Override class priors (default: estimated from data) |

---

## 📊 Naive Bayes vs Other Classifiers

| Aspect | Naive Bayes | Logistic Reg. | Decision Tree | KNN |
|--------|:-----------:|:-------------:|:-------------:|:---:|
| Training Speed | ✅ Very Fast | ✅ Fast | ✅ Fast | ✅ Instant |
| Prediction Speed | ✅ Very Fast | ✅ Very Fast | ✅ Very Fast | ❌ Slow |
| Scales to large n | ✅ Yes | ✅ Yes | ⚠️ Medium | ❌ No |
| Feature Independence | Required | ❌ Not needed | ❌ Not needed | ❌ Not needed |
| Handles missing data | ✅ Yes | ⚠️ With work | ✅ Yes | ❌ No |
| Probabilistic output | ✅ Yes | ✅ Yes | ⚠️ Rough | ⚠️ Rough |
| High-dimensional | ✅ Excellent | ✅ Good | ⚠️ Poor | ❌ Very poor |
| Non-linear boundary | ⚠️ Quadratic | ❌ Linear | ✅ Yes | ✅ Yes |

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Correlated features | Independence assumption violated → overconfident predictions | Use Logistic Regression or Random Forest |
| Zero probabilities | Unseen feature values → entire product = 0 | Use Laplace smoothing (alpha > 0) |
| Non-Gaussian continuous features | GNB assumption violated | Transform features or use different variant |
| Imbalanced classes | Prior dominates prediction | Use ComplementNB or adjust priors |
| Using counts as-is for MultinomialNB | Length bias (longer docs score higher) | Use TF-IDF normalization |

---

## 🔗 Related Topics

- `Logistic_Regression` — Also probabilistic, but without independence assumption
- `Decision_Trees` — Non-linear, no distributional assumptions
- `06_Feature_Selection` — Select independent features to satisfy NB assumption
- `10_Natural_Language_Processing` — MultinomialNB for text classification

---

## 📚 References

- Scikit-learn Naive Bayes: [https://scikit-learn.org/stable/modules/naive_bayes.html](https://scikit-learn.org/stable/modules/naive_bayes.html)
- An Introduction to Statistical Learning — Chapter 4.4
- The Elements of Statistical Learning — Chapter 6.6
- Original spam filtering paper (Sahami et al., 1998)
