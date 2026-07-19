# Halcyon Credit ML Risk Scoring Engine: Decision Threshold Policy

This document establishes the official decision threshold policy for classification of credit risk using the LightGBM champion model. 

The champion pipeline is designed to output a raw continuous probability of default ($P(\text{Default})$). The final credit decision (Approve vs. Reject) is governed by a configurable decision threshold ($T$), where:
- $\text{Decision} = \text{Reject}$ if $P(\text{Default}) \ge T$
- $\text{Decision} = \text{Approve}$ if $P(\text{Default}) < T$

---

## Business Decision Policies

Depending on macroeconomic conditions, risk tolerance, and customer acquisition targets, the credit risk committee can adopt one of three predefined policy thresholds:

### 1. Conservative Policy (High Recall)
- **Threshold ($T$)**: **0.36**
- **Recall**: 85.46%
- **Precision**: 12.67%
- **F1-Score**: 0.2207
- **False Negatives (Toxic Loans Approved)**: 722 (Intercepted 4,243 out of 4,965 defaults)
- **False Positives (Good Clients Rejected)**: 29,249
- **Use Case**: Recommended during economic contractions, high-interest-rate environments, or for high-risk credit tiers where the cost of write-offs is extremely high. This policy minimizes toxic defaults at the expense of loan volumes.

### 2. Balanced Policy (Best F1-Score)
- **Threshold ($T$)**: **0.66**
- **Recall**: 43.04%
- **Precision**: 24.59%
- **F1-Score**: 0.3130
- **False Negatives**: 2,828
- **False Positives**: 6,554
- **Use Case**: Recommended for standard, day-to-day credit scoring. It achieves the mathematically optimal compromise between customer acquisition and write-off prevention.

### 3. Revenue-Friendly Policy (Fewer False Positives)
- **Threshold ($T$)**: **0.75**
- **Recall**: 24.15%
- **Precision**: 32.72%
- **F1-Score**: 0.2779
- **False Negatives**: 3,766
- **False Positives**: 2,465 (Reduces rejections of good clients to **2,465**)
- **Use Case**: Recommended during economic growth periods or when launching marketing campaigns to rapidly grow market share. This policy maximizes approval rates and customer acquisition, accepting higher write-off rates.

---

## Comparison Table of Policies

| Policy | Threshold | Recall | Precision | F1-Score | Approved Defaults (FN) | Rejected Good Clients (FP) |
|---|---|---|---|---|---|---|
| **Conservative** | 0.36 | 85.5% | 12.7% | 0.2207 | **722** | 29,249 |
| **Balanced** | 0.66 | 43.0% | 24.6% | 0.3130 | 2,828 | 6,554 |
| **Revenue-Friendly** | 0.75 | 24.1% | 32.7% | 0.2779 | 3,766 | **2,465** |

---

## Architectural Implementation Guidance

### Configurable Inference API
To prevent threshold hardcoding, the prediction engine must not return hardcoded binary predictions. Rather, the inference client should load the threshold dynamically from a central config file or request payload:

```python
# Recommended API Client Implementation Pattern
def make_credit_decision(probabilities: np.ndarray, policy: str = "balanced") -> np.ndarray:
    # Load threshold mapping from configuration
    threshold_mapping = {
        "conservative": 0.36,
        "balanced": 0.66,
        "revenue_friendly": 0.75
    }
    
    threshold = threshold_mapping.get(policy.lower(), 0.66)
    
    # Decisions: 1 = Reject (high risk of default), 0 = Approve
    decisions = (probabilities >= threshold).astype(int)
    return decisions
```
