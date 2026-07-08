# ML Risk Scoring Model: Baseline Analysis Report

This document presents a detailed evaluation and side-by-side comparison of the baseline Logistic Regression model before and after introducing improved feature engineering for default risk prediction on the Home Credit dataset.

## Executive Summary

Adding domain-specific financial ratios, demographic metrics, external credit aggregates, missing data counters, and document flag aggregators improved performance across **every single metric**.

### Old vs. Improved Baseline Performance

| Metric | Old Baseline (7 Features) | Improved Baseline (27 Features) | Change / Delta | Business Impact |
|---|---|---|---|---|
| **ROC-AUC** | 0.7337 | **0.7441** | **+0.0104** | Better credit risk rank discrimination. |
| **Accuracy** | 0.6821 | **0.6880** | **+0.0059** | Marginally better overall performance. |
| **Precision** | 0.1556 | **0.1597** | **+0.0041** | Fewer false rejections of creditworthy clients. |
| **Recall** | 0.6638 | **0.6721** | **+0.0083** | Successfully intercepted 41 additional default events. |
| **F1-Score** | 0.2522 | **0.2581** | **+0.0059** | Better harmonic balance of precision/recall. |
| **True Negatives (TN)** | 38,658 | **38,977** | **+319** | 319 more good credit clients accepted. |
| **False Positives (FP)** | 17,880 | **17,561** | **-319** | 319 fewer lost revenue opportunities. |
| **False Negatives (FN)** | 1,669 | **1,628** | **-41** | 41 fewer toxic loan defaults approved. |
| **True Positives (TP)** | 3,296 | **3,337** | **+41** | Correctly blocked 41 additional defaulting loans. |

---

## 1. Description of New Features

The following 15 features were engineered and integrated into the model pipeline:

### Financial Ratios
* **`CREDIT_INCOME_RATIO`** (`AMT_CREDIT / AMT_INCOME_TOTAL`): Measures loan size relative to income. Larger loans relative to earnings denote higher leverage.
* **`ANNUITY_INCOME_RATIO`** (`AMT_ANNUITY / AMT_INCOME_TOTAL`): Measures the monthly payment strain. High payments relative to monthly income impact default probability.
* **`CREDIT_ANNUITY_RATIO`** (`AMT_CREDIT / AMT_ANNUITY`): Serves as a proxy for the expected loan term (length of repayment).
* **`CREDIT_GOODS_RATIO`** (`AMT_CREDIT / AMT_GOODS_PRICE`): Represents the loan-to-value ratio. Ratios > 1.0 indicate consumer financed more than the value of the underlying goods.

### Demographic Features
* **`AGE_YEARS`** (`DAYS_BIRTH / -365.25`): Converts age in negative days into intuitive positive years.
* **`EMPLOYMENT_YEARS`** (`DAYS_EMPLOYED / -365.25`): Converts employment tenure to positive years.
* **`CHILDREN_RATIO`** (`CNT_CHILDREN / CNT_FAM_MEMBERS`): Proportions dependents to family size, reflecting cost-of-living constraints.
* **`INCOME_PER_PERSON`** (`AMT_INCOME_TOTAL / CNT_FAM_MEMBERS`): Captures discretionary income capability per household member.

### External Credit Features
* **`EXT_SOURCE_MEAN`**, **`EXT_SOURCE_STD`**, **`EXT_SOURCE_MAX`**, **`EXT_SOURCE_MIN`**: Computes aggregated statistics across the three predictive external credit score sources (`EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`). Captures the overall strength, volatility, and extremes of external ratings.

### Missing Data Features
* **`TOTAL_MISSING_VALUES`**: Total count of missing fields for the applicant. (Can serve as a proxy for application quality or thinner credit history).
* **`MISSING_PERCENTAGE`**: Percentage of missing data relative to all features.

### Document Flags
* **`TOTAL_DOCUMENT_FLAGS`**: Sums all document flag columns (e.g., `FLAG_DOCUMENT_2` through `FLAG_DOCUMENT_21`). Tracks how many verifying documents were submitted.

---

## 2. Class Distribution & Imbalance Analysis

### Target Variable Distribution
- **Non-Default (Class 0)**: 282,686 applications (91.93%)
- **Default (Class 1)**: 24,825 applications (8.07%)

Only **8.07%** of applicants defaulted. While the dataset remains highly imbalanced, adjusting the class weights to `"balanced"` in the Logistic Regression baseline forces the model to focus on identifying the minority class (Default), bringing the Recall up to **67.21%** at the cost of overall accuracy.

![Class Distribution](class_distribution.png)

---

## 3. Confusion Matrix Analysis (Validation Set)

| Actual \ Predicted | Predicted Non-Default (0) | Predicted Default (1) |
|---|---|---|
| **Actual Non-Default (0)** | **38,977** (True Negative) | **17,561** (False Positive) |
| **Actual Default (1)** | **1,628** (False Negative) | **3,337** (True Positive) |

The improved model reduced False Positives by **319** applications and simultaneously reduced False Negatives by **41** applications.

![Confusion Matrix](confusion_matrix.png)

---

## 4. Performance Curves

### ROC Curve
The area under the ROC curve (ROC-AUC) increased from **0.7337** to **0.7441** (+0.0104), reflecting significantly better ranking discrimination between defaults and non-defaults.

![ROC Curve](roc_curve.png)

### Precision-Recall Curve
Due to the massive class imbalance, the PR curve shows that precision declines as recall increases. However, the overall average precision (AP) improved with the new feature set.

![Precision-Recall Curve](precision_recall_curve.png)

---

## 5. Next Steps for Optimization

While the improved features boosted the Logistic Regression baseline, linear models have limitations (e.g., they cannot capture interactive non-linear relationships without manual cross-multiplication, and they are sensitive to collinearity among the ratios).

Our next optimizations should focus on:
1. **Tree-Based Models (LightGBM/XGBoost)**: Implementing gradient boosting to capture non-linear relationships natively.
2. **Feature Selection**: Dropping collinear raw features if we keep their ratios, reducing overfitting risk.
3. **Advanced Group Aggregations**: Merging customer history tables (such as credit bureau records and historic payment schedules) to enrich the feature set.
