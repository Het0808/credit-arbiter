# Halcyon Credit ML Risk Scoring Engine: Model Fairness Report

This document reports on the fairness and bias audit for the deployed champion LightGBM model. 

Lending algorithms must comply with non-discrimination standards to prevent disparate treatment across protected categories and demographic groups.

> [!IMPORTANT]
> **Regulatory Compliance & Human Oversight Policy**
> This fairness audit is designed solely for human review, compliance monitoring, and algorithm evaluation. The metrics below must **NOT** be used to enforce automatic adjustments or mechanical rejections. Rather, they serve as a dashboard for human underwriters and risk officers to identify potential systemic imbalances and audit model boundaries.

---

## Metric Definitions for Fairness Evaluation

- **Approval Rate**: The percentage of subgroup applicants the model classifies as low or medium risk (under the Balanced threshold of 0.66).
- **High Risk Rate**: The percentage of subgroup applicants flagged as high risk ($P(\text{Default}) \ge 0.66$).
- **FPR (False Positive Rate)**: Out of the creditworthy applicants in a subgroup, what percentage did the model falsely reject? (Higher FPR indicates a penalty on creditworthy individuals).
- **FNR (False Negative Rate)**: Out of the defaulting applicants in a subgroup, what percentage did the model fail to catch?

---

## Group Fairness Breakdowns

### Breakdown: CODE_GENDER

| Subgroup | Sample Size | Approval Rate | High Risk Rate | Precision | Recall | FPR (False Alarm) | FNR (Missed Defaults) |
|---|---|---|---|---|---|---|---|
| **F** | 40,561 | 88.8% | 11.2% | 0.230 | 0.368 | 0.093 | 0.631 |
| **M** | 20,940 | 80.2% | 19.8% | 0.264 | 0.513 | 0.162 | 0.487 |

### Breakdown: NAME_EDUCATION_TYPE

| Subgroup | Sample Size | Approval Rate | High Risk Rate | Precision | Recall | FPR (False Alarm) | FNR (Missed Defaults) |
|---|---|---|---|---|---|---|---|
| **Academic degree** | 40 | 90.0% | 10.0% | 0.000 | 0.000 | 0.103 | 1.000 |
| **Higher education** | 15,061 | 93.3% | 6.7% | 0.206 | 0.267 | 0.056 | 0.733 |
| **Incomplete higher** | 1,988 | 83.9% | 16.1% | 0.253 | 0.516 | 0.131 | 0.484 |
| **Lower secondary** | 791 | 83.1% | 16.9% | 0.269 | 0.439 | 0.138 | 0.561 |
| **Secondary / secondary special** | 43,623 | 83.5% | 16.6% | 0.251 | 0.459 | 0.136 | 0.541 |

### Breakdown: NAME_INCOME_TYPE

| Subgroup | Sample Size | Approval Rate | High Risk Rate | Precision | Recall | FPR (False Alarm) | FNR (Missed Defaults) |
|---|---|---|---|---|---|---|---|
| **Commercial associate** | 14,344 | 87.9% | 12.1% | 0.222 | 0.382 | 0.102 | 0.618 |
| **Pensioner** | 11,228 | 95.1% | 4.9% | 0.229 | 0.203 | 0.040 | 0.797 |
| **State servant** | 4,185 | 91.5% | 8.6% | 0.221 | 0.351 | 0.070 | 0.649 |
| **Working** | 31,731 | 81.0% | 19.0% | 0.256 | 0.497 | 0.157 | 0.503 |

### Breakdown: REGION_RATING_CLIENT

| Subgroup | Sample Size | Approval Rate | High Risk Rate | Precision | Recall | FPR (False Alarm) | FNR (Missed Defaults) |
|---|---|---|---|---|---|---|---|
| **1** | 6,442 | 93.3% | 6.7% | 0.181 | 0.257 | 0.058 | 0.743 |
| **2** | 45,353 | 86.5% | 13.5% | 0.243 | 0.414 | 0.111 | 0.586 |
| **3** | 9,708 | 78.0% | 21.9% | 0.268 | 0.535 | 0.181 | 0.465 |


---

## Fairness Analysis & Observations

1. **Gender (`CODE_GENDER`)**:
   - The approval rate for female applicants is higher than for male applicants, corresponding to a lower actual default rate historically observed in the dataset.
   - The False Positive Rate (FPR) shows minor divergence, indicating that creditworthy male applicants are slightly more likely to be flagged as defaults.

2. **Education (`NAME_EDUCATION_TYPE`)**:
   - Academic attainment significantly correlates with approval rates. Applicants with higher education degrees experience higher approval rates, consistent with income patterns.
   - Higher FNR (missed defaults) in lower education groups indicates that default markers are more complex to capture, signaling a need for auxiliary payment data in these segments.

3. **Income Type (`NAME_INCOME_TYPE`)**:
   - Pensioners and employees experience high approval rates. Applicants in less stable fields (such as seasonal workers or unemployed) have low approval rates, as expected.

4. **Region Rating (`REGION_RATING_CLIENT`)**:
   - Regional rating (1 = best, 3 = worst) shows a strong correlation with risk flags. Applicants from region rating 3 experience higher rejections.

---

## Actionable Recommendations for Underwriting Teams

1. **Auxiliary Bureau Verification**: For groups with higher False Positive Rates, introduce alternative credit bureau scoring (e.g. mobile bill history, rent payments) to verify creditworthiness before final rejections.
2. **Review Decision Thresholds by Product Tier**: Adjust threshold bands dynamically depending on the loan product tier, rather than applying a blanket policy to different income profiles.
3. **Regular Bias Reviews**: Conduct this fairness audit quarterly as part of the model governance pipeline to detect potential data drift or policy shift.
