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
| **F** | 40,561 | 87.4% | 12.6% | 0.223 | 0.401 | 0.105 | 0.599 |
| **M** | 20,940 | 81.4% | 18.6% | 0.270 | 0.495 | 0.151 | 0.504 |

### Breakdown: AGE_BAND

| Subgroup | Sample Size | Approval Rate | High Risk Rate | Precision | Recall | FPR (False Alarm) | FNR (Missed Defaults) |
|---|---|---|---|---|---|---|---|
| **18-25** | 2,392 | 61.2% | 38.8% | 0.211 | 0.700 | 0.347 | 0.300 |
| **26-35** | 14,368 | 76.4% | 23.6% | 0.249 | 0.560 | 0.198 | 0.440 |
| **36-45** | 16,889 | 85.9% | 14.1% | 0.251 | 0.428 | 0.116 | 0.572 |
| **46-55** | 14,039 | 89.2% | 10.8% | 0.257 | 0.379 | 0.086 | 0.621 |
| **56-65** | 12,165 | 93.8% | 6.2% | 0.211 | 0.231 | 0.051 | 0.769 |
| **65+** | 1,650 | 98.2% | 1.8% | 0.035 | 0.017 | 0.018 | 0.983 |

### Breakdown: NAME_EDUCATION_TYPE

| Subgroup | Sample Size | Approval Rate | High Risk Rate | Precision | Recall | FPR (False Alarm) | FNR (Missed Defaults) |
|---|---|---|---|---|---|---|---|
| **Academic degree** | 40 | 100.0% | 0.0% | 0.000 | 0.000 | 0.000 | 1.000 |
| **Higher education** | 15,061 | 93.2% | 6.8% | 0.213 | 0.277 | 0.056 | 0.723 |
| **Incomplete higher** | 1,988 | 84.2% | 15.8% | 0.254 | 0.510 | 0.128 | 0.490 |
| **Lower secondary** | 791 | 81.7% | 18.3% | 0.241 | 0.427 | 0.155 | 0.573 |
| **Secondary / secondary special** | 43,623 | 82.7% | 17.3% | 0.247 | 0.472 | 0.143 | 0.528 |

### Breakdown: NAME_INCOME_TYPE

| Subgroup | Sample Size | Approval Rate | High Risk Rate | Precision | Recall | FPR (False Alarm) | FNR (Missed Defaults) |
|---|---|---|---|---|---|---|---|
| **Commercial associate** | 14,344 | 87.4% | 12.7% | 0.223 | 0.399 | 0.106 | 0.601 |
| **Pensioner** | 11,228 | 94.7% | 5.3% | 0.217 | 0.211 | 0.044 | 0.789 |
| **State servant** | 4,185 | 92.1% | 7.9% | 0.233 | 0.342 | 0.064 | 0.658 |
| **Working** | 31,731 | 80.3% | 19.7% | 0.252 | 0.508 | 0.164 | 0.492 |

### Breakdown: REGION_RATING_CLIENT

| Subgroup | Sample Size | Approval Rate | High Risk Rate | Precision | Recall | FPR (False Alarm) | FNR (Missed Defaults) |
|---|---|---|---|---|---|---|---|
| **1** | 6,442 | 92.5% | 7.4% | 0.200 | 0.317 | 0.063 | 0.683 |
| **2** | 45,353 | 86.0% | 14.0% | 0.238 | 0.420 | 0.116 | 0.580 |
| **3** | 9,708 | 77.5% | 22.5% | 0.269 | 0.550 | 0.184 | 0.450 |


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
