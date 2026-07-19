# Model Comparison: Baseline Logistic Regression vs. LightGBM Challenger

This report compares the performance of the baseline Logistic Regression model (v1) and the LightGBM challenger model (v1) on the Home Credit Default Risk dataset using identical datasets and preprocessing features.

## Side-by-Side Performance Comparison

| Metric | Baseline (Logistic Regression v1) | Challenger (LightGBM v1) | Delta (Challenger - Baseline) | Business Impact |
|---|---|---|---|---|
| **ROC-AUC** | 0.7415 | **0.7617** | **+0.0203** | **Significant improvement** in risk ranking. |
| **Accuracy** | 0.6856 | **0.7047** | **+0.0191** | Overall predictive accuracy increased. |
| **Precision** | 0.1577 | **0.1696** | **+0.0119** | **Fewer false alarms**; fewer good clients rejected. |
| **Recall** | 0.6669 | **0.6824** | **+0.0155** | Intercepts more defaulting loan applications. |
| **F1-Score** | 0.2551 | **0.2717** | **+0.0166** | Better harmonic balance of risk metrics. |

### Confusion Matrix Delta

| Prediction Category | Baseline (LogReg v1) | Challenger (LightGBM v1) | Count Delta | Business Impact |
|---|---|---|---|---|
| **True Negatives (TN)** | 38,855 | 39,953 | **+1098** | Correctly approved more creditworthy applications. |
| **False Positives (FP)** | 17,683 | 16,585 | **-1098** | Reduced false alarms (fewer lost clients). |
| **False Negatives (FN)** | 1,654 | 1,577 | **-77** | Reduced toxic leakage (fewer unpaid defaults). |
| **True Positives (TP)** | 3,311 | 3,388 | **+77** | Intercepted more defaults. |

---

## Performance Analysis & Insights

1. **ROC-AUC Performance**:
   - The LightGBM model achieves a ROC-AUC of **0.7617**, outperforming the Logistic Regression baseline by **0.0203**.
   - Because gradient boosted trees model non-linear boundaries natively, LightGBM handles non-linear feature interactions (such as the interaction between `EXT_SOURCE` fields and financial ratios) far better than the baseline linear regression.

2. **Precision and Recall Trade-off**:
   - LightGBM increases **Precision** by **+0.0119** and increases **Recall** by **+0.0155**.
   - In most business cases, moving one metric compromises the other. However, because LightGBM has greater overall discriminative power, it shifts the entire frontier outward, resulting in a simultaneous increase in both the capture rate (Recall) and the efficiency rate (Precision).

3. **Financial Impact**:
   - By switching to LightGBM, the credit team blocks more default events (FN decreased), saving major default capital, and approves more creditworthy candidates (FP decreased), securing extra interest margins.

---

## Recommendation
Based on the metrics, the **LightGBM v1 challenger model is recommended to replace the Logistic Regression model** as the active champion model in the Halcyon Credit Risk Scoring Engine.
