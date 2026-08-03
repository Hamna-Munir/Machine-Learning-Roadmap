# 📘 Classification Metrics — Theory

---

## 📌 What are Classification Metrics?

Classification metrics quantify how well a model assigns data points to the  
correct discrete categories. Unlike regression, predictions are compared against  
**true class labels** rather than continuous values — but the right metric depends  
heavily on class distribution, costs of different errors, and business context.

```
Binary Classification:       Multiclass:
  ŷ ∈ {0, 1}                  ŷ ∈ {0, 1, 2, ..., K-1}

  Ground truth comparison:
    Correct predictions  → True Positives / True Negatives
    Wrong predictions    → False Positives / False Negatives
```

---

## 🧮 The Confusion Matrix — Foundation of All Metrics

For binary classification (positive class = 1, negative class = 0):

```
                     PREDICTED
                  Positive   Negative
ACTUAL  Positive │   TP    │   FN   │
        Negative │   FP    │   TN   │

TP (True Positive) : Predicted Positive, Actually Positive ✅
TN (True Negative) : Predicted Negative, Actually Negative ✅
FP (False Positive): Predicted Positive, Actually Negative ❌ (Type I Error)
FN (False Negative): Predicted Negative, Actually Positive ❌ (Type II Error)

Total = TP + TN + FP + FN = n (all samples)
```

**Memory device:**
```
The FIRST word tells you what the model predicted:
  True  → model was CORRECT
  False → model was WRONG

The SECOND word tells you what the model predicted:
  Positive → model predicted POSITIVE
  Negative → model predicted NEGATIVE
```

---

## 📐 Core Classification Metrics

---

### 1. Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
         = Correct predictions / Total predictions

Properties:
  ✅ Intuitive — fraction of correct predictions
  ✅ Good when classes are balanced
  ❌ MISLEADING on imbalanced data

Classic trap:
  Dataset: 950 negatives, 50 positives
  Predict ALL negative → Accuracy = 95%!
  But model never detects a single positive.

When to use:
  → Only when classes are roughly balanced
  → Never as the sole metric on imbalanced data

sklearn: accuracy_score(y_true, y_pred)
```

### 2. Precision

```
Precision = TP / (TP + FP)

"Of all predictions of POSITIVE, how many were actually positive?"

Example:
  Spam detection: Precision = 0.95 means
  → 95% of emails flagged as spam were actually spam
  → Only 5% of spam flags were false alarms

High Precision → model rarely cries wolf (few false alarms)
Low Precision  → many false positives (too many false alarms)

When to use:
  → When False Positives are costly
  → Spam filters, fraud alerts (don't want to flag legitimate emails/transactions)

sklearn: precision_score(y_true, y_pred)
```

### 3. Recall (Sensitivity / True Positive Rate)

```
Recall = TP / (TP + FN)

"Of all ACTUAL positives, how many did the model find?"

Example:
  Cancer detection: Recall = 0.95 means
  → 95% of actual cancer cases were detected
  → Only 5% of cancers were missed

High Recall → model misses very few positives
Low Recall  → many positives are missed (False Negatives)

When to use:
  → When False Negatives are costly
  → Medical diagnosis, fraud detection (missing a case is dangerous)

sklearn: recall_score(y_true, y_pred)
```

### 4. Precision-Recall Trade-off

```
There is always a trade-off between Precision and Recall:

Threshold ↑ → Fewer positives predicted
  → Precision ↑  (fewer false alarms)
  → Recall ↓    (miss more positives)

Threshold ↓ → More positives predicted
  → Precision ↓  (more false alarms)
  → Recall ↑    (catch more positives)

Finding the right threshold depends on the cost of FP vs FN:
  High cost of FP (false alarm) → push threshold UP → prioritize Precision
  High cost of FN (miss)        → push threshold DOWN → prioritize Recall

Default sklearn threshold: 0.5 (often suboptimal!)
```

### 5. F1 Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = 2TP / (2TP + FP + FN)

Harmonic mean of Precision and Recall.
Penalizes extreme imbalances between them.

F1 = 1.0 → Perfect Precision AND Recall
F1 = 0.0 → Either Precision or Recall is 0

Properties:
  ✅ Balances Precision and Recall
  ✅ Good for imbalanced classes
  ✅ Doesn't require knowing TN (useful when TN is trivially large)
  ❌ Treats FP and FN as equally costly

When to use:
  → Default metric for imbalanced classification
  → When both Precision and Recall matter equally

sklearn: f1_score(y_true, y_pred)
```

### 6. Fβ Score

```
Fβ = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)

β controls the trade-off:
  β = 1   → Equal weight (same as F1)
  β = 2   → Recall weighted 2× more than Precision (β > 1 → prioritize Recall)
  β = 0.5 → Precision weighted 2× more than Recall (β < 1 → prioritize Precision)

Use F2 when missing positives (FN) is more costly.
Use F0.5 when false alarms (FP) are more costly.

sklearn: fbeta_score(y_true, y_pred, beta=2)
```

### 7. Specificity (True Negative Rate)

```
Specificity = TN / (TN + FP)

"Of all actual negatives, how many did the model correctly reject?"

Complement of False Positive Rate:
  Specificity = 1 − FPR

High Specificity → very few false alarms
→ Used alongside Recall/Sensitivity in medical contexts

sklearn: No direct function; compute as:
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  specificity = tn / (tn + fp)
```

### 8. ROC-AUC (Area Under the ROC Curve)

```
ROC Curve: plots True Positive Rate (Recall) vs False Positive Rate
           at every possible threshold

FPR = FP / (FP + TN) = 1 − Specificity

AUC (Area Under Curve):
  AUC = 1.0  → Perfect classifier
  AUC = 0.5  → Random classifier (diagonal line)
  AUC < 0.5  → Worse than random (flip predictions!)

Properties:
  ✅ Threshold-independent — evaluates model at ALL thresholds
  ✅ Scale-free (range: 0 to 1)
  ✅ Measures separability of positive and negative classes
  ❌ Can be optimistic on heavily imbalanced data
  ❌ Averages over all thresholds — may not reflect chosen threshold

Interpretation:
  AUC = 0.80 → 80% of the time, model ranks a random positive
               ABOVE a random negative

When to use:
  → Comparing models regardless of threshold
  → Evaluating ranking/scoring quality
  → Balanced or mildly imbalanced datasets

sklearn: roc_auc_score(y_true, y_prob)
         # Note: needs probabilities, not predictions!
```

### 9. Precision-Recall AUC (PR-AUC)

```
PR Curve: plots Precision vs Recall at every threshold

PR-AUC:
  PR-AUC = 1.0 → Perfect Precision at all Recall levels
  PR-AUC ≈ class prevalence → Random classifier

Properties:
  ✅ More informative than ROC-AUC for heavily imbalanced data
  ✅ Focuses only on the positive class
  ✅ Not affected by large TN counts (unlike ROC)
  ❌ Harder to interpret than AUC

Thumb rule:
  Balanced data        → ROC-AUC is fine
  Imbalanced data      → PR-AUC is more informative

sklearn: average_precision_score(y_true, y_prob)
         precision_recall_curve(y_true, y_prob)
```

### 10. Log Loss (Binary Cross-Entropy)

```
Log Loss = −(1/n) × Σ [yᵢ × log(p̂ᵢ) + (1−yᵢ) × log(1−p̂ᵢ)]

Where p̂ᵢ = predicted probability of positive class

Properties:
  ✅ Penalizes confident wrong predictions heavily
  ✅ Rewards well-calibrated probability estimates
  ✅ Directly optimized by logistic regression and neural networks
  ❌ Requires probabilities (not just class predictions)
  ❌ Sensitive to poorly calibrated probabilities
  ❌ No upper bound (lower = better, 0 = perfect)

sklearn: log_loss(y_true, y_prob)
```

### 11. Matthews Correlation Coefficient (MCC)

```
MCC = (TP×TN − FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]

Properties:
  ✅ Ranges from −1 to +1 (+1 = perfect, 0 = random, −1 = inverse)
  ✅ Works well even with extremely imbalanced classes
  ✅ Accounts for ALL four cells of confusion matrix
  ✅ Considered the most informative single binary classification metric
  ❌ Less commonly reported (less familiar to stakeholders)

MCC = +1 → perfect predictions
MCC =  0 → random guessing
MCC = −1 → perfect inverse predictions (flip all predictions)

sklearn: matthews_corrcoef(y_true, y_pred)
```

### 12. Cohen's Kappa

```
κ = (p_o − p_e) / (1 − p_e)

Where:
  p_o = observed accuracy
  p_e = expected accuracy by chance

Properties:
  ✅ Corrects for class imbalance by accounting for chance agreement
  ✅ Range: (−1, 1] — higher is better
  ❌ More complex to interpret than accuracy

κ > 0.8 → strong agreement
κ > 0.6 → moderate agreement
κ < 0.2 → poor agreement

sklearn: cohen_kappa_score(y_true, y_pred)
```

---

## 🔢 Multiclass Classification

For K > 2 classes, binary metrics are extended via averaging:

```
Averaging strategies:

macro:     Compute metric per class, take unweighted mean
           → Treats all classes equally regardless of size
           → Good when small classes matter

weighted:  Compute metric per class, weight by class support (size)
           → Accounts for class imbalance
           → Closer to accuracy for imbalanced data

micro:     Aggregate TP/FP/FN globally, then compute metric
           → Dominated by majority class
           → Equivalent to accuracy for most metrics

none:      Return metric per class as array
           → Inspect each class individually

sklearn:   f1_score(y_true, y_pred, average='macro')
           f1_score(y_true, y_pred, average='weighted')
           f1_score(y_true, y_pred, average='micro')
```

---

## 📊 Metric Selection Guide

```
Balanced dataset?
  → Accuracy is meaningful
  → ROC-AUC for threshold-independent evaluation

Imbalanced dataset?
  → F1 / F2 / F0.5 instead of accuracy
  → PR-AUC over ROC-AUC
  → MCC for single-number summary

Cost of False Positive > Cost of False Negative?
  → Prioritize Precision (e.g. spam filter)
  → Use F0.5 or threshold > 0.5

Cost of False Negative > Cost of False Positive?
  → Prioritize Recall (e.g. cancer detection)
  → Use F2 or threshold < 0.5

Need probability calibration?
  → Log Loss

Need to compare ranking quality?
  → ROC-AUC / PR-AUC

Most comprehensive single metric?
  → MCC (binary) or Macro-F1 (multiclass)
```

---

## 🎛️ Threshold Tuning

```
Default threshold = 0.5 is rarely optimal.

Steps:
  1. Train model and get predicted probabilities
  2. Compute Precision, Recall, F1 at each threshold
  3. Choose threshold that maximizes your chosen metric

For Precision-Recall trade-off:
  from sklearn.metrics import precision_recall_curve
  precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

For ROC:
  from sklearn.metrics import roc_curve
  fpr, tpr, thresholds = roc_curve(y_true, y_prob)

Youden's J statistic: threshold that maximizes (TPR − FPR)
  best_thresh = thresholds[np.argmax(tpr - fpr)]
```

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Using accuracy on imbalanced data | 95% accuracy on 95:5 data is trivial | Use F1, PR-AUC, or MCC |
| ROC-AUC on severe imbalance | AUC optimistic due to large TN | Use PR-AUC instead |
| Using 0.5 threshold always | Suboptimal for most real problems | Tune threshold on validation set |
| Ignoring confusion matrix | Aggregate metrics hide per-class failures | Always inspect the confusion matrix |
| Reporting only one metric | Single metrics miss important trade-offs | Report Precision, Recall, F1, AUC |
| Averaging over classes for imbalanced data | Macro-F1 ignores class sizes | Use weighted-F1 or report per-class |

---

## 🆚 Metric Quick Reference

| Metric | Formula | Range | Lower Better? | Needs Probabilities? |
|--------|---------|-------|:-------------:|:-------------------:|
| Accuracy | (TP+TN)/n | [0,1] | ❌ | ❌ |
| Precision | TP/(TP+FP) | [0,1] | ❌ | ❌ |
| Recall | TP/(TP+FN) | [0,1] | ❌ | ❌ |
| F1 | 2PR/(P+R) | [0,1] | ❌ | ❌ |
| ROC-AUC | Area under ROC | [0,1] | ❌ | ✅ |
| PR-AUC | Area under PR | [0,1] | ❌ | ✅ |
| Log Loss | Cross-entropy | [0,∞) | ✅ | ✅ |
| MCC | Correlation | [−1,1] | ❌ | ❌ |
| Cohen's κ | Chance-corrected | (−1,1] | ❌ | ❌ |

---

## 🔗 Related Topics

- `05_Model_Evaluation/regression_metrics.md` — MAE, RMSE, R² for regression
- `05_Model_Evaluation/cross_validation.md` — Stratified K-Fold, Leave-One-Out
- `06_Feature_Selection/` — Use metric scores to evaluate feature subsets
- `07_Hyperparameter_Tuning/` — Scoring parameter in GridSearchCV
- `Logistic_Regression` — Classification probabilities and threshold tuning
- `Random_Forest`, `XGBoost` — Feature importance + classification metrics

---

## 📚 References

- Scikit-learn Classification Metrics: [https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- ROC and AUC explained: [https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
- MCC paper (Chicco & Jurman, 2020): "The advantages of the Matthews correlation coefficient"
- An Introduction to Statistical Learning — Chapter 4
- The Elements of Statistical Learning — Chapter 7
