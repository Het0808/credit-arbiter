# Hyperparameter Tuning Metrics Comparison

This document records the metrics comparison for the Halcyon Credit Risk Scoring Engine across model development phases. The GridSearchCV-tuned LightGBM model has been established as the production champion.

## Performance Metrics Table

The table below contrasts the model performance on the test split (61,503 records, 20% partition of the dataset):

| Model Stage | Model Type | ROC-AUC | Accuracy | Precision | Recall (Default Capture) | F1-Score |
|---|---|---|---|---|---|---|
| **Baseline v1** | Logistic Regression | 0.7441 | 68.80% | 15.97% | 67.21% | 0.2581 |
| **Challenger v1** | LightGBM (Untuned) | 0.7617 | 70.21% | 16.85% | 68.38% | 0.2704 |
| **Randomized Search** | LightGBM (Random Tuned) | 0.7642 | 69.95% | 16.78% | 68.78% | 0.2698 |
| **Champion v1 (Grid Search)** | **LightGBM (Grid Tuned)** | **0.7654** | **70.41%** | **17.04%** | **68.88%** | **0.2732** |

---

## Detailed Observations

1. **ROC-AUC (Primary Metric)**:
   * The grid-tuned LightGBM model achieved the highest overall discriminative performance with a **0.7654 ROC-AUC**, representing a **+0.0213** net increase over the baseline Logistic Regression and a **+0.0037** increase over the untuned LightGBM model.
2. **Recall (Default Capture Rate)**:
   * The grid-tuned model successfully captures **68.88%** of defaulting borrowers, representing the highest protection against bad loans. This is a **+1.67%** recall improvement over the baseline (saving 83 default events from being approved).
3. **Confusion Matrix Counts (Test Set)**:
   * **Baseline v1**: True Negatives (TN): 38,977 | False Positives (FP): 17,561 | False Negatives (FN): 1,628 | True Positives (TP): 3,337
   * **Challenger v1**: TN: 39,788 | FP: 16,750 | FN: 1,570 | TP: 3,395
   * **Randomized Search**: TN: 39,607 | FP: 16,931 | FN: 1,550 | TP: 3,415
   * **Grid Search Champion**: TN: **39,883** | FP: **16,655** | FN: **1,545** | TP: **3,420**

---

## Deployed Parameters for Tuned Champion
The production pipeline is loaded with these optimized parameters found by GridSearchCV:
* `learning_rate`: `0.05`
* `max_depth`: `10`
* `n_estimators`: `200`
* `subsample`: `0.8` (held constant)
* `colsample_bytree`: `0.8` (held constant)
