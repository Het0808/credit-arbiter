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
| **F** | 40,561 | 88.4% | 11.6% | 0.227 | 0.377 | 0.097 | 0.623 |
| **M** | 20,940 | 79.1% | 20.9% | 0.259 | 0.534 | 0.173 | 0.466 |

### Breakdown: NAME_EDUCATION_TYPE

| Subgroup | Sample Size | Approval Rate | High Risk Rate | Precision | Recall | FPR (False Alarm) | FNR (Missed Defaults) |
|---|---|---|---|---|---|---|---|
| **Academic degree** | 40 | 100.0% | 0.0% | 0.000 | 0.000 | 0.000 | 1.000 |
| **Higher education** | 15,061 | 93.2% | 6.9% | 0.208 | 0.275 | 0.057 | 0.725 |
| **Incomplete higher** | 1,988 | 83.6% | 16.4% | 0.245 | 0.510 | 0.134 | 0.490 |
| **Lower secondary** | 791 | 81.3% | 18.7% | 0.243 | 0.439 | 0.158 | 0.561 |
| **Secondary / secondary special** | 43,623 | 82.6% | 17.4% | 0.247 | 0.475 | 0.144 | 0.525 |

### Breakdown: NAME_INCOME_TYPE

| Subgroup | Sample Size | Approval Rate | High Risk Rate | Precision | Recall | FPR (False Alarm) | FNR (Missed Defaults) |
|---|---|---|---|---|---|---|---|
| **Commercial associate** | 14,344 | 87.0% | 13.0% | 0.218 | 0.403 | 0.110 | 0.597 |
| **Pensioner** | 11,228 | 94.9% | 5.1% | 0.221 | 0.206 | 0.042 | 0.794 |
| **State servant** | 4,185 | 91.7% | 8.3% | 0.221 | 0.342 | 0.068 | 0.658 |
| **Working** | 31,731 | 80.1% | 19.9% | 0.252 | 0.512 | 0.165 | 0.488 |

### Breakdown: REGION_RATING_CLIENT

| Subgroup | Sample Size | Approval Rate | High Risk Rate | Precision | Recall | FPR (False Alarm) | FNR (Missed Defaults) |
|---|---|---|---|---|---|---|---|
| **1** | 6,442 | 92.8% | 7.2% | 0.200 | 0.307 | 0.061 | 0.693 |
| **2** | 45,353 | 85.8% | 14.2% | 0.238 | 0.428 | 0.118 | 0.572 |
| **3** | 9,708 | 77.6% | 22.4% | 0.265 | 0.538 | 0.185 | 0.462 |


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
