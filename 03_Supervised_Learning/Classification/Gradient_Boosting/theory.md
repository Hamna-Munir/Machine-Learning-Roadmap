# 📘 Gradient Boosting — Theory

---

## 📌 What is Gradient Boosting?

Gradient Boosting is an **ensemble method** that builds models **sequentially** —  
each new model corrects the **residual errors** of the previous ensemble.  
Unlike Random Forest (parallel, independent trees), Gradient Boosting trains  
trees **one after another**, each fitting the gradient of the loss function.

```
Initial prediction (mean/constant)
     │
     ▼
  Residuals_1 = y − ŷ₀
     │
     ▼
  Tree_1 fits Residuals_1
     │
     ▼
  ŷ₁ = ŷ₀ + η × Tree_1(x)         (η = learning rate)
     │
     ▼
  Residuals_2 = y − ŷ₁
     │
     ▼
  Tree_2 fits Residuals_2
     │
     ▼
  ...repeat for N trees...
     │
     ▼
  Final: ŷ = ŷ₀ + η×Tree_1 + η×Tree_2 + ... + η×Tree_N
```

> 💡 "Gradient Boosting says: 'I know I'm wrong — let me figure out exactly  
>     HOW I'm wrong and fix it step by step.'"

---

## 🔍 When to Use Gradient Boosting?

| Condition | Use Gradient Boosting? |
|-----------|:----------------------:|
| Tabular / structured data | ✅ Yes — excellent |
| Maximum accuracy needed | ✅ Yes |
| Medium-sized dataset (1K–500K) | ✅ Yes |
| Time available for tuning | ✅ Yes |
| Need fast training with large data | ❌ No → Use LightGBM |
| Need interpretability | ❌ No → Use Decision Tree |
| Sequential/streaming data | ❌ No |

---

## 🧮 The Gradient Boosting Algorithm

### General Framework

```
1. Initialize model with a constant: F₀(x) = argmin_γ Σ L(yᵢ, γ)

   For m = 1, 2, ..., M (each tree):

2. Compute pseudo-residuals (negative gradient of loss):
     rᵢₘ = − [∂L(yᵢ, F(xᵢ)) / ∂F(xᵢ)]   for i = 1..n

3. Fit a regression tree hₘ(x) to the pseudo-residuals {rᵢₘ}

4. Find the optimal step size (leaf values) γⱼₘ:
     γⱼₘ = argmin_γ Σ L(yᵢ, Fₘ₋₁(xᵢ) + γ)
              xᵢ in leaf j

5. Update the model:
     Fₘ(x) = Fₘ₋₁(x) + η × Σⱼ γⱼₘ × 𝟙[x ∈ Rⱼₘ]

6. Return FM(x) as final model
```

### Why "Gradient"?

```
In Gradient Descent for optimization:
  θ ← θ − α × ∇L(θ)    (move in direction of steepest descent)

In Gradient Boosting for function space:
  F(x) ← F(x) − α × ∇L(F)   (add a function that reduces the loss)

Each tree approximates the GRADIENT of the loss function:
  → For squared error loss: pseudo-residuals = actual residuals (yᵢ − ŷᵢ)
  → For log loss: pseudo-residuals = yᵢ − p̂ᵢ  (actual − predicted probability)
  → For absolute error: pseudo-residuals = sign(yᵢ − ŷᵢ)
```

---

## 📐 Loss Functions

| Task | Loss Function | Pseudo-Residual |
|------|:------------:|:---------------:|
| Regression | Squared Error (L2) | yᵢ − ŷᵢ |
| Regression | Absolute Error (L1) | sign(yᵢ − ŷᵢ) |
| Regression | Huber | Blends L1 and L2 |
| Classification (binary) | Log Loss | yᵢ − p̂ᵢ |
| Classification (multi) | Softmax | Generalized log loss |

---

## 🎛️ Key Hyperparameters

### The Three Core Parameters

```
1. n_estimators (M): Number of trees
   More trees → better fit → but slower + risk of overfitting without regularization
   Typical: 100–1000 (use early stopping)

2. learning_rate (η): Shrinkage — scales each tree's contribution
   Smaller η → more trees needed → better generalization
   Typical: 0.01–0.3
   Rule: Decrease η → Increase n_estimators proportionally

3. max_depth: Depth of each individual tree
   Shallow trees (max_depth=3–5) → weak learners → standard GBM
   Typical: 3–5 (deeper trees = more complex interactions captured)
```

### Additional Regularization Parameters

```
min_samples_split  : Min samples to split a node
min_samples_leaf   : Min samples per leaf
max_features       : Features per split (adds randomness like Random Forest)
subsample          : Fraction of training samples per tree (stochastic GBM)
                     subsample < 1.0 → Stochastic Gradient Boosting
                     Adds randomness → reduces variance → often helps
```

---

## ⚖️ Learning Rate vs N_Estimators Tradeoff

```
High learning rate + few trees:
  Fast training, but coarse optimization, may miss optimal

Low learning rate + many trees:
  Slow training, but fine optimization, typically better generalization

Rule of thumb:
  Start: learning_rate=0.1, n_estimators=100
  Tune:  learning_rate=0.05, n_estimators=200  (same budget, often better)
  Fine:  learning_rate=0.01, n_estimators=1000 (best, but slow)

sklearn: GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3
)
```

---

## 📊 Stochastic Gradient Boosting

Adds **row subsampling** — each tree is trained on a random fraction of data:

```
subsample=1.0  → Standard GBM (use all data for each tree)
subsample=0.8  → Use 80% of training data randomly (without replacement)
subsample=0.5  → 50% — higher variance reduction, but more randomness

Benefits:
  ✅ Reduces overfitting (variance reduction)
  ✅ Speeds up training
  ✅ Provides OOB-like estimation
  ✅ Allows feature importance via OOB score

Also: max_features controls feature subsampling per split
  max_features='sqrt' → Same as Random Forest feature bagging
```

---

## 📈 Bias-Variance in Gradient Boosting

```
Early stopping at few trees:
  High Bias    — not enough boosting rounds
  Low Variance — simple model

Too many trees without regularization:
  Low Bias     — fits training data well
  High Variance — overfits if learning_rate is large

Optimal zone:
  Achieved through:
  1. Small learning_rate (η ≤ 0.1)
  2. Subsampling (subsample < 1.0)
  3. Shallow trees (max_depth = 3–5)
  4. Early stopping or sufficient n_estimators

Train Error              Test Error
│●                       │            ●●●●●●
│ ●●                     │        ●●●●
│   ●●●●●●●●●●●●●        │    ●●●●
│                        │  ●●
└──────────── n_trees     └──────────── n_trees
  (always drops)           (U-shaped — find minimum)
```

---

## 🌿 Feature Importance in GBM

```
Same as Decision Tree importance:
  Σ (weighted impurity reduction) at all nodes using this feature
  Averaged across all trees in the ensemble

Normalized to sum = 1.0

sklearn: model.feature_importances_

More reliable: Use permutation importance on test set
```

---

## 🆚 GBM vs Random Forest vs XGBoost

| Aspect | Gradient Boosting | Random Forest | XGBoost |
|--------|:-----------------:|:-------------:|:-------:|
| Learning | Sequential | Parallel | Sequential (optimized) |
| Trees | Weak learners (shallow) | Full-depth | Weak learners (shallow) |
| Variance | ✅ Low (if tuned) | ✅ Low | ✅ Low |
| Bias | ✅ Low | ⚠️ Medium | ✅ Very Low |
| Speed | ❌ Slower | ✅ Fast | ✅ Fast |
| Overfitting Risk | ⚠️ High without tuning | ✅ Low | ⚠️ Medium |
| Regularization | ⚠️ Limited | ✅ Built-in | ✅ L1 + L2 built-in |
| Accuracy | ✅ Very Good | ✅ Good | ✅ Excellent |

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Too many trees + large learning_rate | Severe overfitting | Use small η (0.05–0.1) + more trees |
| max_depth too large | Complex trees = high variance | Keep max_depth = 3–5 |
| No subsampling | Full data per tree = correlated trees | Set subsample=0.8 |
| No validation monitoring | Can't detect overfitting | Use validation_fraction + n_iter_no_change |
| Default n_estimators=100 | Often too few for small learning_rate | Tune together with learning_rate |
| Skipping feature importance check | Missing signal about data quality | Always check feature importances |

---

## 🔗 Related Topics

- `Random_Forest` — Parallel ensemble alternative (lower variance, less tuning)
- `XGBoost` — Optimized GBM with L1+L2 regularization
- `LightGBM` — Histogram-based GBM, much faster
- `CatBoost` — GBM with native categorical support
- `08_Ensemble_Learning` — Full boosting vs bagging comparison
- `07_Hyperparameter_Tuning` — GridSearchCV for GBM hyperparameters

---

## 📚 References

- Scikit-learn `GradientBoostingClassifier`: [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- Scikit-learn `HistGradientBoostingClassifier`: [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
- Original GBM Paper (Friedman, 2001): [https://www.jstor.org/stable/2699986](https://www.jstor.org/stable/2699986)
- An Introduction to Statistical Learning — Chapter 8.2.3
- The Elements of Statistical Learning — Chapter 10
