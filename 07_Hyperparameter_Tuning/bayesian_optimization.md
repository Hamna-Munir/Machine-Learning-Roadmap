# 📘 Bayesian Optimization — Hyperparameter Tuning Theory

---

## 📌 What is Bayesian Optimization?

Bayesian Optimization is a **sequential, model-based strategy** for finding the  
hyperparameter configuration that minimizes (or maximizes) an expensive black-box  
objective function — using as **few evaluations as possible**.

```
Black-box objective f(λ):
  λ = hyperparameter configuration  (e.g. learning_rate=0.05, max_depth=6)
  f(λ) = CV validation score        (expensive — requires full model training)

Goal: find λ* = argmax f(λ) with as few f(λ) evaluations as possible

Bayesian Optimization:
  Builds a SURROGATE MODEL of f(λ) from past evaluations
  Uses ACQUISITION FUNCTION to decide where to evaluate next
  → Balances exploration (unknown regions) vs exploitation (known good regions)
```

---

## 🔍 Why Bayesian Over Grid / Random Search?

```
Grid Search:
  → Exhaustively tries all combinations
  → O(Kⁿ) evaluations (K values per n hyperparameters)
  → Very slow for large search spaces
  → Wastes evaluations in bad regions

Random Search:
  → Randomly samples configurations
  → Better than grid for high-d spaces (Bergstra & Bengio, 2012)
  → Still wastes evaluations — no learning from past results

Bayesian Optimization:
  → Learns from past evaluations to focus on promising regions
  → Typically needs 10–50× fewer evaluations than grid search
  → Ideal when each evaluation is expensive (deep learning, large ensembles)
  → Handles continuous, discrete, and conditional hyperparameters
```

**Sample efficiency comparison:**
```
Evaluations needed to find near-optimal config:

Grid Search     : ████████████████████████████████████  ~1000 evals
Random Search   : ████████████████████  ~200 evals
Bayesian Optim  : ████  ~40 evals

(Approximate, depends on search space dimensionality)
```

---

## 🧮 The Three Components

### 1. Surrogate Model — Gaussian Process (GP)

```
A Gaussian Process models the objective function f(λ) as a distribution
over functions — giving both a PREDICTED MEAN and UNCERTAINTY at every point.

GP(λ) ~ N(μ(λ), σ²(λ))

Where:
  μ(λ)  = predicted mean score at hyperparameter config λ
  σ²(λ) = predicted uncertainty (variance) at λ

Properties:
  ✅ Provides uncertainty estimates (crucial for exploration)
  ✅ Exact Bayesian posterior update after each observation
  ✅ Works well for low-d search spaces (< 20 hyperparameters)
  ❌ Scales as O(n³) — slow for many observations (> 1000)
  ❌ Struggles with high-dimensional search spaces (> 20 params)

Alternative surrogates:
  Tree Parzen Estimator (TPE)   → used by Optuna, Hyperopt (scales better)
  Random Forest surrogate       → used by SMAC (handles categorical well)
  Neural network surrogate      → used by BOHB
```

### 2. Acquisition Function — Where to Sample Next

```
The acquisition function uses μ(λ) and σ(λ) to decide where to evaluate next.
It balances exploration (high σ) vs exploitation (high μ).

Common acquisition functions:

Expected Improvement (EI):           ← most common
  EI(λ) = E[max(f(λ) − f*, 0)]
  f* = best observed value so far
  → Picks λ with highest expected gain over the current best

Upper Confidence Bound (UCB):
  UCB(λ) = μ(λ) + κ × σ(λ)
  κ controls exploration-exploitation tradeoff
  → Higher κ → more exploration of uncertain regions

Probability of Improvement (PI):
  PI(λ) = P(f(λ) > f* + ξ)
  → Tends to be greedy (exploitation-heavy)

EI is the default in most libraries — good balance for most use cases.
```

### 3. Update Loop

```
Algorithm:

  1. Initialize: evaluate f(λ) at n_initial random configurations
  2. Fit surrogate model on all (λ, f(λ)) observations so far
  3. Maximize acquisition function → λ_next (next candidate)
  4. Evaluate f(λ_next) → expensive ML training + CV
  5. Add (λ_next, f(λ_next)) to observations
  6. Repeat 2–5 for n_calls evaluations
  7. Return λ* = argmax observed f(λ)
```

---

## 📦 Python Libraries

### scikit-optimize (skopt)

```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

search_space = {
    'n_estimators'  : Integer(50, 500),
    'max_depth'     : Integer(2, 15),
    'learning_rate' : Real(0.001, 0.3, prior='log-uniform'),
    'subsample'     : Real(0.5, 1.0),
}

opt = BayesSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    search_spaces=search_space,
    n_iter=50,           # number of Bayesian evaluations
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
)
opt.fit(X_train, y_train)
print(opt.best_params_, opt.best_score_)
```

### Optuna (Tree Parzen Estimator)

```python
import optuna

def objective(trial):
    params = {
        'n_estimators' : trial.suggest_int('n_estimators', 50, 500),
        'max_depth'    : trial.suggest_int('max_depth', 2, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'subsample'    : trial.suggest_float('subsample', 0.5, 1.0),
    }
    model  = GradientBoostingClassifier(**params, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(study.best_params)
print(study.best_value)
```

### Hyperopt

```python
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

space = {
    'n_estimators' : hp.quniform('n_estimators', 50, 500, 50),
    'max_depth'    : hp.quniform('max_depth', 2, 15, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.3)),
    'subsample'    : hp.uniform('subsample', 0.5, 1.0),
}

def objective(params):
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth']    = int(params['max_depth'])
    model  = GradientBoostingClassifier(**params, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    return {'loss': -scores.mean(), 'status': STATUS_OK}

trials = Trials()
best   = fmin(objective, space, algo=tpe.suggest,
              max_evals=50, trials=trials)
```

---

## 🎛️ Search Space Design

```
Parameter types:

Real (continuous):
  learning_rate : Real(0.001, 0.3, prior='log-uniform')  ← use log for rates
  subsample     : Real(0.5, 1.0)

Integer (discrete):
  n_estimators  : Integer(50, 500)
  max_depth     : Integer(2, 15)

Categorical:
  criterion     : Categorical(['gini', 'entropy'])
  optimizer     : Categorical(['adam', 'sgd', 'rmsprop'])

Conditional (Optuna):
  if trial.suggest_categorical('use_dropout', [True, False]):
      dropout = trial.suggest_float('dropout', 0.1, 0.5)

Log-uniform prior:
  Use for: learning_rate, regularization (alpha, C, lambda)
  Reason: these parameters matter more in magnitude than absolute value
          → equal probability in log space: [0.001, 0.01, 0.1] not [0.001, 0.5, 0.999]
```

---

## 🔄 Bayesian vs Grid vs Random — When to Use Each

| Aspect | Grid Search | Random Search | Bayesian |
|--------|:-----------:|:-------------:|:--------:|
| Sample efficiency | ❌ Poor | ⚠️ Medium | ✅ Best |
| Handles high-d spaces | ❌ No | ✅ Yes | ⚠️ Up to ~20 params |
| Parallelizable | ✅ Fully | ✅ Fully | ⚠️ Partially |
| Handles conditional params | ❌ No | ❌ No | ✅ Yes |
| Implementation complexity | ✅ Simple | ✅ Simple | ⚠️ Moderate |
| Best for | Small spaces, few params | First exploration | Expensive objectives |
| sklearn native | ✅ GridSearchCV | ✅ RandomizedSearchCV | ❌ Needs skopt/Optuna |

**Decision rule:**
```
n_params ≤ 3 AND budget unlimited   → Grid Search
n_params > 3 OR budget limited      → Random Search (baseline)
Evaluation expensive (> 1 min each) → Bayesian Optimization
Need conditional hyperparameters    → Optuna (TPE)
```

---

## 📊 Convergence and Early Stopping

```
Convergence plot:
  Plot best observed score vs number of evaluations

Score
│                    ●────────────────── (converged)
│               ●────
│          ●────
│      ●───
│  ●───
│●
└──────────────────────── Evaluations
   ↑              ↑
   Exploration    Exploitation
   (high variance) (fine-tuning)

Early stopping: stop when improvement < ε for last N evaluations
  skopt:  callback=DeltaYStopper(delta=0.001, n_best=10)
  Optuna: study.optimize(objective, n_trials=50, callbacks=[...])
```

---

## ⚠️ Common Pitfalls

| Pitfall | Issue | Fix |
|--------|-------|-----|
| Too few initial random points | Surrogate poorly initialized | Use n_initial_points ≥ 10 |
| Search space too wide | Slow convergence | Start broad, then refine |
| Search space too narrow | Miss optimal region | Include wide range initially |
| Not using log-uniform for rates | Under-samples small values | Always log-uniform for LR, α, C |
| Evaluating on test set during tuning | Data leakage → optimistic | Use CV on train only |
| Running too few iterations | Under-converged | Minimum 30–50 evaluations |
| Not fixing random_state | Non-reproducible results | Fix seed in study/BayesSearchCV |
| Ignoring warm_start / caching | Re-trains from scratch each time | Use caching where possible |

---

## 🔗 Related Topics

- `07_Hyperparameter_Tuning/grid_random_search.md` — Baseline tuning methods
- `07_Hyperparameter_Tuning/hyperparameter_tuning.ipynb` — Full tuning notebook
- `05_Model_Evaluation/cross_validation.ipynb` — CV inside tuning loop
- `08_Ensemble_Learning/` — Ensemble hyperparameters to tune
- `XGBoost`, `LightGBM`, `CatBoost` — Primary candidates for Bayesian tuning

---

## 📚 References

- Scikit-optimize (skopt): [https://scikit-optimize.github.io/](https://scikit-optimize.github.io/)
- Optuna: [https://optuna.readthedocs.io/](https://optuna.readthedocs.io/)
- Hyperopt: [http://hyperopt.github.io/hyperopt/](http://hyperopt.github.io/hyperopt/)
- Original Bayesian Optimization Paper (Mockus, 1975)
- Bergstra & Bengio (2012): "Random Search for Hyper-Parameter Optimization"
- Snoek et al. (2012): "Practical Bayesian Optimization of Machine Learning Algorithms"
- An Introduction to Statistical Learning — Chapter 5 (Resampling Methods)
