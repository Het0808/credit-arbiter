# Halcyon Credit Risk Scoring: Model Comparison Report

This report presents a side-by-side performance evaluation and comparison of all three machine learning models trained for default risk prediction in the Halcyon Credit Arbiter engine:

1. **Logistic Regression Baseline**
2. **LightGBM Champion (Grid Tuned)**
3. **XGBoost Challenger**

---

## 1. Performance Comparison Table

All models are evaluated on the identical unseen 20% test partition (61,503 records):

| Model | ROC-AUC | Accuracy | Precision | Recall (Default Capture) | F1-Score | False Positives (FP) | False Negatives (FN) | Training Time (s) |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression Baseline** | 0.7441 | 68.80% | 15.97% | 67.21% | 0.2581 | 17,561 | 1,628 | ~3.2 |
| **LightGBM Champion (Grid Tuned)** | 0.7654 | 70.41% | 17.04% | 68.88% | 0.2732 | 16,655 | 1,545 | ~12.4 |
| **XGBoost Challenger** | 0.7655 | 72.67% | 17.83% | 66.14% | 0.2809 | 15,130 | 1,681 | 2.61 |

---

## 2. Performance Rankings

Models are ranked in order of priority: 
1. **ROC-AUC** (discriminative power)
2. **Recall** (default capture rate)
3. **F1-Score** (harmonic balance)

### Official Ranking Table:
1. **XGBoost Challenger** (ROC-AUC: 0.7655 | Recall: 66.14% | F1: 0.2809)
2. **LightGBM Champion (Grid Tuned)** (ROC-AUC: 0.7654 | Recall: 68.88% | F1: 0.2732)
3. **Logistic Regression Baseline** (ROC-AUC: 0.7441 | Recall: 67.21% | F1: 0.2581)

**The Deployed Champion is identified as: `XGBoost Challenger`.**

---

## 3. Deep-Dive Analysis

### Why the Champion Model is Better
The champion model (`XGBoost Challenger`) outperforms the alternatives due to several structural factors:
1. **Non-linear Multi-feature Interactions**: Credit default risk is rarely linear. A borrower's risk is determined by composite situations (e.g. moderate income combined with a very high credit limit *and* a low external source rating). Tree-based architectures capture these high-order interactions naturally, whereas Logistic Regression requires manual feature engineering of cross-product terms.
2. **Missing Data Handling**: Tree-based algorithms (LightGBM and XGBoost) handle missing values natively by routing missing indices to the optimal split branch during training. The baseline Logistic Regression relies on median imputation, which shifts feature distributions and distorts signal quality.
3. **Algorithmic Optimizations**: 
   - **XGBoost** uses L1/L2 regularization in the tree splitting objective and handles gradient/hessian calculations to construct robust level-wise splits.
   - **LightGBM** uses leaf-wise (best-first) tree growth, which often results in deeper, more accurate trees on tabular credit datasets. 
   - XGBoost's level-wise tree growth with early stopping on validation ROC-AUC has generalized exceptionally well here, avoiding the overfitting ceiling and scoring peak ROC-AUC.

### Business Impact
Evaluating credit models relies heavily on the financial trade-off between **False Negatives (FN)** (approving a defaulter, resulting in average loan principal loss) and **False Positives (FP)** (rejecting a creditworthy applicant, resulting in interest margin loss).

* **Cost of Toxic Default (FN)**: Assuming an average default loss of **$15,000** per applicant.
* **Cost of False Rejection (FP)**: Assuming an average lost interest margin of **$1,500** per applicant.

Let's calculate the financial loss on the test split for each model:

* **Logistic Regression Baseline**:
  * Default Loss: 1,628 × $15,000 = **$24,420,000**
  * Opportunity Loss: 17,561 × $1,500 = **$26,341,500**
  * **Total Credit Loss: $50,761,500**
  
* **LightGBM Champion (Grid Tuned)**:
  * Default Loss: 1,545 × $15,000 = **$23,175,000**
  * Opportunity Loss: 16,655 × $1,500 = **$24,982,500**
  * **Total Credit Loss: $48,157,500**
  
* **XGBoost Challenger**:
  * Default Loss: 1,681 × $15,000 = **$25,215,000**
  * Opportunity Loss: 15,130 × $1,500 = **$22,695,000**
  * **Total Credit Loss: $47,910,000**

**Financial Insight**: XGBoost achieves the lowest total credit loss ($47.91M), saving **$247,500** relative to LightGBM and **$2,851,500** relative to the Logistic Regression baseline. It achieves this by significantly reducing False Positives (FP) by 1,525 applications, which outweighs the cost of the 136 additional False Negatives (FN).

### Engineering Tradeoffs
1. **Model Representation**: Logistic Regression is represented by 32 weights and 1 intercept, which is extremely lightweight and fast to serialize. LightGBM and XGBoost require loading ensemble structures containing hundreds of trees.
2. **Library Dependencies**: Logistic Regression utilizes standard `scikit-learn`. Incorporating `lightgbm` and `xgboost` requires installing compiled C++ wrapper libraries in the runtime Docker container, adding setup complexity.
3. **Robustness to Scale**: LightGBM is highly optimized for memory footprint. XGBoost, while extremely powerful, can consume substantial RAM during fitting on larger tables.

### Computational Cost
1. **Training Latency**: XGBoost trains in **2.61s** using the histogram-based method and early stopping. This is faster than LightGBM (~12.4s) and comparable to Logistic Regression (~3.2s).
2. **Inference Latency**: Logistic Regression has a sub-microsecond prediction latency (a simple dot product). Tree ensembles take between 2ms to 10ms to traverse hundreds of trees, which is still highly acceptable for online decision-making (typical SLAs are <100ms).

### Explainability
* **Logistic Regression**: High explainability. Coefficients represent direct log-odds weights. Regulators favor this for credit auditing because the reason for rejection is transparent.
* **LightGBM and XGBoost**: Black-box models. While we can compute global Feature Importances, explaining individual loan rejections requires secondary mathematical frameworks such as SHAP (Shapley Additive exPlanations) or LIME, adding operational complexity.

---

## 4. Production Recommendation

We recommend deploying the **`XGBoost Challenger`** model into production.

### Justification:
1. **Peak Discriminative Power**: XGBoost achieves the highest test ROC-AUC (**0.7655**), outperforming LightGBM (**0.7654**) and Logistic Regression (**0.7441**).
2. **Lowest Total Credit Cost**: XGBoost saves the business the most capital on the test set ($47.91M vs LightGBM's $48.16M and LogReg's $50.76M), representing a **+$247,500** net improvement over the prior champion.
3. **Training Speed**: The XGBoost model utilizing `tree_method="hist"` and early stopping trained in just **2.61 seconds**, demonstrating excellent efficiency.
4. **Generalization**: The use of early stopping prevented the model from hitting the feature information capacity ceiling, yielding a well-regularized ensemble.
