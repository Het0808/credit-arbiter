# Halcyon Credit ML Risk Scoring Engine: Baseline Model Report

This document registers the official v1 baseline credit risk prediction model for the Halcyon Credit Arbiter project.

## Model Description
The model is a **Logistic Regression** pipeline, regularized and configured to handle substantial class imbalances. It integrates data imputation, numerical scaling, categorical encoding, and domain-engineered features inside a single, unified scikit-learn Pipeline object.

- **Algorithm**: Logistic Regression (`max_iter=1000`, `class_weight="balanced"`)
- **Pipeline version**: `v1`
- **Training Timestamp**: 2026-07-08 11:20:05

## Dataset Used
- **Source**: Home Credit Default Risk application training dataset (`application_train.csv`).
- **Total Records**: 307,511 applications.
- **Training Records (80%)**: 246,008 applications.
- **Test Records (20%)**: 61,503 applications.

## Target & Features Used
- **Target Variable**: `TARGET` (1 for defaulted applicant, 0 for non-defaulted applicant)
- **Total Feature Count**: 29 features (numerical + categorical after encoding).

### Numeric Features List
- `AMT_INCOME_TOTAL`
- `AMT_CREDIT`
- `AMT_ANNUITY`
- `DAYS_EMPLOYED`
- `EXT_SOURCE_1`
- `EXT_SOURCE_2`
- `EXT_SOURCE_3`
- `CNT_FAM_MEMBERS`
- `AMT_GOODS_PRICE`
- `CNT_CHILDREN`
- `CREDIT_INCOME_RATIO`
- `ANNUITY_INCOME_RATIO`
- `CREDIT_ANNUITY_RATIO`
- `CREDIT_GOODS_RATIO`
- `EMPLOYMENT_YEARS`
- `CHILDREN_RATIO`
- `INCOME_PER_PERSON`
- `EXT_SOURCE_MEAN`
- `EXT_SOURCE_STD`
- `EXT_SOURCE_MAX`
- `EXT_SOURCE_MIN`
- `TOTAL_MISSING_VALUES`
- `MISSING_PERCENTAGE`
- `TOTAL_DOCUMENT_FLAGS`

### Categorical Features List
- `NAME_CONTRACT_TYPE`
- `FLAG_OWN_CAR`
- `FLAG_OWN_REALTY`
- `NAME_INCOME_TYPE`
- `NAME_EDUCATION_TYPE`

## Feature Engineering Summary
The following features were engineered to capture financial stress, applicant demographics, rating volatility, and record quality:
1. **Financial Ratios**:
   - `CREDIT_INCOME_RATIO`: Loan size relative to income (leverage indicator).
   - `ANNUITY_INCOME_RATIO`: Monthly repayment obligation relative to income.
   - `CREDIT_ANNUITY_RATIO`: Repayment duration proxy.
   - `CREDIT_GOODS_RATIO`: Loan-to-value ratio for consumer goods.
2. **Demographic Features**:
   - `AGE_YEARS`: Age in years.
   - `EMPLOYMENT_YEARS`: Employment tenure in years.
   - `CHILDREN_RATIO`: Ratio of children to total family members.
   - `INCOME_PER_PERSON`: Average discretionary income per household member.
3. **External Credit Features**:
   - `EXT_SOURCE_MEAN`, `EXT_SOURCE_STD`, `EXT_SOURCE_MAX`, `EXT_SOURCE_MIN`: Summarized indicators of external bureau rankings.
4. **Missing Data Features**:
   - `TOTAL_MISSING_VALUES`, `MISSING_PERCENTAGE`: Counts showing record completeness.
5. **Document Flags**:
   - `TOTAL_DOCUMENT_FLAGS`: Total count of submitted verification forms.

## Evaluation Metrics

| Metric | Baseline v1 Value |
|---|---|
| **ROC-AUC** | **0.7415** |
| **Accuracy** | **0.6856** |
| **Precision** | **0.1577** |
| **Recall** | **0.6669** |
| **F1-Score** | **0.2551** |

### Confusion Matrix (Test Set)
- **True Negatives (TN)**: 38,855 (Creditworthy applicants approved)
- **False Positives (FP)**: 17,683 (Creditworthy applicants rejected)
- **False Negatives (FN)**: 1,654 (Defaulting applicants approved)
- **True Positives (TP)**: 3,311 (Defaulting applicants blocked)

## Visualizations

### Confusion Matrix
![Confusion Matrix](plots/confusion_matrix.png)

### ROC Curve
![ROC Curve](plots/roc_curve.png)

### Precision-Recall Curve
![Precision-Recall Curve](plots/precision_recall_curve.png)

## Business Interpretation
- **Default Capture Rate (Recall)**: The model intercepts **66.69%** of defaulting applications. Intercepting defaults directly protects capital.
- **Approval Quality (Precision)**: Because of the low precision (15.77%), flagging an applicant as default has a relatively high false alarm rate. For every true default blocked, the model flags approximately 5 non-defaulting applications. While this trades off customer acquisition, it is standard in conservative risk profiles.
- **ROC-AUC (0.7415)**: The model possesses strong ranking capability, which is suitable for tiering interest rates and limits.

## Known Limitations
1. **Linear Assumptions**: Logistic Regression assumes linear log-odds relations, failing to model compound interactive effects (e.g. high credit limit *low* income).
2. **Imputation Bias**: Standard median imputation distorts distribution shapes.
3. **Information Collinearity**: Financial ratios derived from the same base columns lead to regression collinearity.

## Recommendations for Next Iteration
1. **Transition to Tree-Based Ensemble (LightGBM/XGBoost)**: To capture interactive dependencies without hand-crafting interactions.
2. **Auxiliary Relational Data**: Incorporate bureau credit histories (`bureau.csv`) and previous applications (`previous_application.csv`).
3. **Advanced Threshold Tuning**: Set the classification threshold dynamically based on the dollar cost of false negatives vs. false positives rather than defaulting to 0.5.
