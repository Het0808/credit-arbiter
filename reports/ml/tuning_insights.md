# Diagnostic Insights: Model Performance Ceiling & Tuning Dynamics

This document provides a technical diagnostic analyzing why the expanded hyperparameter grid search (evaluated on a subset and refitted) did not out-perform the production champion model.

---

## 1. Scale Dependency of Hyperparameters (Subset vs. Full Data)

Hyperparameters are highly dependent on the **volume of training data**. A configuration that is optimal for a small sample size can become sub-optimal when scaled up to a much larger dataset.

* **Tuned Model Search Regime**: To execute 375 cross-validation fits quickly, the search ran on a **20,000 stratified subset** (8% of the data).
  * **Result**: The search selected a slow learning rate (`0.01`) and a deep tree structure (`max_depth=12`, `n_estimators=250`).
  * **Behavior on Full Data**: When refitted on the full 246,008 training rows, the slow learning rate (`0.01`) did not allow the model to fully learn the complex interactions before reaching the 250-tree limit (underfitting), or conversely, the depth of `12` caused the model to overfit to localized noise.
* **Production Champion Regime**: Tunan directly on the **full training dataset** (using the smaller 8-combination grid).
  * **Result**: Selected `learning_rate=0.05`, `max_depth=8`, and `n_estimators=150`, which generalized better on the test set, achieving **0.7654 ROC-AUC**.

---

## 2. Overfitting in the Expanded Parameter Grid

Looking at the bottom of the grid results (lines 112 to 126 in [grid_search_all_results.csv](file:///Users/het08/Desktop/Credit-arbiter/credit-arbiter/reports/ml/grid_search_all_results.csv)), we can observe classic signs of model degradation:
* **Worst Combinations**: Learning rate = `0.1`, max depth = `10` to `12`, and estimators = `200` to `250` produced CV ROC-AUC scores below **0.6950**.
* **Reason**: Deep trees combined with higher learning rates and many estimators allow gradient boosting to overfit. The model memorized individual samples in the 20,000 search subset, resulting in poor generalisation.

---

## 3. The Feature Information Capacity Ceiling

In machine learning, algorithms hit an information ceiling based on the features they are given:
* **The Constraint**: Our MVP model uses **32 features** (e.g. basic income ratios, age, and external source averages).
* **The Ceiling**: We have likely extracted the maximum predictive power possible from these 32 variables. Further hyperparameter tuning yields diminishing returns (changes of $\pm 0.001$).
* **The Solution**: To increase the ROC-AUC score significantly (e.g. towards `0.7800+`), we must engineering new predictive features from relational databases, rather than tuning classifier trees.

---

## Actionable Strategy to Increase Metrics

To break the current performance ceiling, we recommend shifting focus from hyperparameter tuning to **feature enrichment**:

1. **Incorporate Relational Credit History**:
   * Merge `bureau.csv` to calculate the number of active loans, total debt ratios, and historical credit inquiries.
   * Merge `previous_application.csv` to capture historical default flags or loan reject history with Home Credit.
2. **Increase Hyperparameter Search Sample Size**:
   * If running parameter sweeps, increase the search subset from 20,000 to **100,000 samples** (running on a cluster or letting it run overnight) to align search parameters closer to the full data regime.
3. **Use Early Stopping**:
   * Implement early stopping in LightGBM (`early_stopping_rounds=10`) during tuning. This allows the model to dynamically stop adding trees when validation metrics plateau, preventing overfitting.
