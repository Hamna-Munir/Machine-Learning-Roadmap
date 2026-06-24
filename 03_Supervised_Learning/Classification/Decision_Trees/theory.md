# 📘 Decision Trees — Theory

---

## 📌 What is a Decision Tree?

A Decision Tree is a **tree-structured model** that makes predictions by learning a  
sequence of **if-else decision rules** inferred from the training data — recursively  
splitting the dataset into purer and purer subsets based on feature thresholds.

```
                    [Age < 30?]
                   /            \
                 Yes             No
                 /                \
        [Income < 50K?]      [Has_Degree?]
         /          \           /        \
       Yes          No        Yes        No
        |            |          |          |
   [No Churn]   [Churn]    [No Churn]  [Churn]
```

> 💡 "A Decision Tree asks a sequence of yes/no questions about the data,  
>      narrowing down to a prediction — just like a flowchart."

---

## 🔍 When to Use Decision Trees?

| Condition | Use Decision Trees? |
|-----------|:-------------------:|
| Need interpretability / explainability | ✅ Yes — primary strength |
| Mixed categorical and numerical features | ✅ Yes — no encoding needed |
| Non-linear decision boundary | ✅ Yes |
| No time for feature scaling | ✅ Yes — scale-invariant |
| Need probability calibration | ⚠️ Caution — probabilities are rough |
| Want a robust, low-variance model | ❌ No → Use Random Forest |
| Very high-dimensional sparse data | ❌ No → Use Naive Bayes / Logistic Regression |

---

## 🧮 Building the Tree — Recursive Splitting

```
Algorithm (CART - Classification and Regression Trees):

1. Start with all data at the root node
2. For every feature and every possible threshold:
     a. Split data into two groups (left, right)
     b. Compute the "impurity" of each split
3. Choose the split that MOST REDUCES impurity
4. Repeat recursively on each child node
5. Stop when:
     - Max depth reached
     - Minimum samples per leaf reached
     - No further impurity reduction possible
     - Node is pure (all samples same class)
```

---

## 📐 Splitting Criteria

### 1. Gini Impurity (Default in sklearn)

Measures the probability of misclassifying a randomly chosen element:

```
Gini(node) = 1 − Σ pᵢ²
                  i

Where pᵢ = proportion of class i in the node

Range: 0 (pure node, all same class) to 0.5 (binary, perfectly mixed)

Example:
  Node with 80% Class A, 20% Class B:
  Gini = 1 − (0.8² + 0.2²) = 1 − (0.64 + 0.04) = 0.32

  Pure node (100% Class A):
  Gini = 1 − (1.0² + 0.0²) = 0.0  ← perfectly pure
```

### 2. Entropy (Information Gain)

Measures the disorder/uncertainty in a node:

```
Entropy(node) = − Σ pᵢ × log₂(pᵢ)
                   i

Range: 0 (pure node) to 1 (binary, perfectly mixed — log₂(2)=1)

Information Gain = Entropy(parent) − Σ (nᵢ/n) × Entropy(childᵢ)

→ The split with the HIGHEST information gain is chosen
```

**Gini vs Entropy:**
```
Both produce similar trees in practice.
Gini is slightly faster to compute (no log).
Entropy is more sensitive to changes in class probabilities.

sklearn default: criterion='gini'
```

### 3. For Regression Trees — MSE / MAE

```
MSE(node) = (1/n) × Σ (yᵢ − ȳ)²    ← variance of target in the node

→ Split that most reduces the weighted average MSE of children is chosen
→ Leaf prediction = mean(y) of all samples in that leaf
```

---

## 🌳 Tree Structure Terminology

```
Root Node       → Top of the tree, contains all data
Internal Node   → A decision point (split on a feature)
Leaf Node       → Terminal node, holds the final prediction
Depth           → Number of levels from root to deepest leaf
Branch          → Path from one node to another
```

---

## 🎛️ Key Hyperparameters

| Parameter | Effect | Typical Values |
|-----------|--------|----------------|
| `max_depth` | Maximum tree depth | 3–10 (None = unlimited, risks overfitting) |
| `min_samples_split` | Min samples required to split a node | 2–20 |
| `min_samples_leaf` | Min samples required in a leaf | 1–10 |
| `max_features` | Max features considered per split | None, 'sqrt', 'log2' |
| `criterion` | Splitting metric | 'gini', 'entropy' (clf); 'squared_error' (reg) |
| `max_leaf_nodes` | Max number of leaf nodes | None or integer |
| `min_impurity_decrease` | Min impurity reduction to allow a split | 0.0+ |

---

## ✂️ Pruning — Controlling Overfitting

Decision Trees easily **overfit** by growing too deep and memorizing training data.

### Pre-Pruning (Early Stopping)
```
Stop splitting BEFORE the tree becomes too complex:
  - max_depth = 5          → limits how deep the tree can grow
  - min_samples_split = 10 → requires enough samples to justify a split
  - min_samples_leaf = 5   → requires enough samples per leaf
```

### Post-Pruning (Cost Complexity Pruning)
```
Grow the FULL tree first, then prune back branches that don't improve
generalization, using a complexity parameter alpha (ccp_alpha):

Cost = Σ Impurity(leaf) + α × (number of leaves)

Larger α → more aggressive pruning → simpler tree

sklearn: DecisionTreeClassifier(ccp_alpha=0.01)
Use cost_complexity_pruning_path() to find optimal alpha via CV
```

---

## 📊 Bias-Variance in Decision Trees

```
Shallow tree (max_depth=2):
  High Bias    — misses important patterns
  Low Variance — stable across different training sets

Deep tree (max_depth=None):
  Low Bias     — captures fine-grained patterns
  High Variance — unstable, overfits, sensitive to noise

           Train Accuracy          Test Accuracy
Depth=2        0.75                    0.74
Depth=5        0.89                    0.86
Depth=10       0.97                    0.83   ← overfitting begins
Depth=None     1.00                    0.79   ← severe overfitting
```

---

## 🎯 Feature Importance

Decision Trees naturally compute feature importance based on  
**how much each feature reduces impurity** across all splits:

```
Importance(feature) = Σ (weighted impurity decrease at nodes using this feature)
                       all nodes

Normalized so all importances sum to 1.0

sklearn: model.feature_importances_
```

---

## 📐 Decision Boundary

Decision Trees create **axis-aligned, rectangular decision regions**:

```
  x₂
  │ ┌─────┬───────┐
  │ │  A  │   B   │
  │ ├─────┼───┬───┤
  │ │  A  │ B │ A │
  │ └─────┴───┴───┘
  └──────────────── x₁

Each split is a straight line perpendicular to one axis (x₁ or x₂)
→ Cannot represent diagonal boundaries efficiently
→ Many splits needed for boundaries like x₁ = x₂ (diagonal line)
```

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| No max_depth set | Tree memorizes training data | Always set max_depth or min_samples_leaf |
| Using accuracy on imbalanced trees | Misleading | Use F1, precision, recall |
| Comparing feature importance across runs | Unstable with small data changes | Use Random Forest for stable importance |
| Ignoring class_weight | Biased toward majority class | Use class_weight='balanced' |
| Deep trees without pruning | High variance, poor generalization | Use ccp_alpha or limit depth |
| Treating tree as final model | Single trees are unstable | Use ensembles (Random Forest, Boosting) |

---

## 🆚 Decision Tree vs Other Classifiers

| Aspect | Decision Tree | Random Forest | Logistic Reg. | SVM |
|--------|:-------------:|:-------------:|:-------------:|:---:|
| Interpretability | ✅ Very High | ⚠️ Medium | ✅ High | ❌ Low |
| Variance | ❌ High | ✅ Low | ✅ Low | ✅ Low |
| Feature Scaling | ❌ Not needed | ❌ Not needed | ✅ Required | ✅ Required |
| Non-linear Boundary | ✅ Yes | ✅ Yes | ❌ No | ✅ (kernel) |
| Handles Categoricals | ✅ Natively | ✅ Natively | ⚠️ Needs encoding | ⚠️ Needs encoding |
| Training Speed | ✅ Fast | ⚠️ Medium | ✅ Fast | ❌ Slow |

---

## 🔗 Related Topics

- `Random_Forest` — Ensemble of Decision Trees via bagging
- `Gradient_Boosting` — Sequential ensemble of shallow trees
- `XGBoost` / `LightGBM` / `CatBoost` — Optimized boosting frameworks
- `06_Feature_Selection` — Tree-based feature importance ranking
- `07_Hyperparameter_Tuning` — GridSearchCV for max_depth, min_samples_split

---

## 📚 References

- Scikit-learn `DecisionTreeClassifier`: [https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- Scikit-learn Decision Trees Guide: [https://scikit-learn.org/stable/modules/tree.html](https://scikit-learn.org/stable/modules/tree.html)
- CART Algorithm (Breiman et al., 1984)
- An Introduction to Statistical Learning — Chapter 8
- The Elements of Statistical Learning — Chapter 9.2
