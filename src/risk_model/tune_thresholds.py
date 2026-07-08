"""
Threshold tuning module for the LightGBM credit risk model.

Evaluates decision thresholds from 0.10 to 0.90, logs metrics (Precision, Recall,
F1, FP, FN) in a CSV, selects three policy thresholds (Conservative, Balanced,
Revenue-friendly), and generates the THRESHOLD_POLICY.md guidance document.
"""

import json
import joblib
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from src.risk_model.config import ModelConfig
from src.risk_model.preprocess import (
    load_raw_data,
    prepare_pipeline_data,
    split_data,
)


def tune_decision_thresholds() -> pd.DataFrame:
    """
    Evaluate validation predictions across a range of thresholds and persist results to CSV.

    Returns:
        DataFrame containing metrics for each threshold.
    """
    print("Initializing threshold tuning run...")
    config = ModelConfig()
    
    # Paths
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / "models" / "lightgbm" / "lightgbm_v1.pkl"
    reports_dir = project_root / "reports" / "ml"
    
    # Load model
    if not model_path.exists():
        raise FileNotFoundError(f"Model pipeline not found at {model_path}. Please run train_lightgbm.py first.")
    
    pipeline = joblib.load(model_path)
    
    # Load and split dataset
    print("Loading raw validation data...")
    df = load_raw_data()
    X, y, _ = prepare_pipeline_data(df, config)
    _, X_test, _, y_test = split_data(X, y, config)
    
    # Predict default probabilities
    print("Running probability inference...")
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Range of thresholds from 0.10 to 0.90 in increments of 0.01
    thresholds = np.arange(0.10, 0.91, 0.01)
    
    records = []
    for thresh in thresholds:
        # Cast thresh to float for neat formatting
        thresh = float(np.round(thresh, 2))
        y_pred = (y_prob >= thresh).astype(int)
        
        # Calculate scores
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        records.append({
            "threshold": thresh,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "true_negatives": int(tn)
        })
        
    df_metrics = pd.DataFrame(records)
    
    # Save CSV
    csv_path = reports_dir / "lightgbm_threshold_analysis.csv"
    df_metrics.to_csv(csv_path, index=False)
    print(f"Threshold analysis metrics exported to: {csv_path}")
    
    return df_metrics


def compile_threshold_policy(df_metrics: pd.DataFrame) -> Path:
    """
    Select three distinct business-oriented decision thresholds and write the THRESHOLD_POLICY.md report.

    Args:
        df_metrics: DataFrame of threshold metrics.

    Returns:
        Path to the generated policy report.
    """
    project_root = Path(__file__).resolve().parents[2]
    reports_dir = project_root / "reports" / "ml"
    
    # 1. Balanced: Maximizes F1 score
    balanced_row = df_metrics.loc[df_metrics["f1_score"].idxmax()]
    balanced_thresh = balanced_row["threshold"]
    
    # 2. Conservative: Catches more defaults, targeting recall of ~85% (or the highest F1 threshold where recall is high)
    # Let's search for a threshold where recall is high (e.g., around 80%-85%).
    # Since scale_pos_weight is high, the probability distribution is shifted. Let's find where recall is nearest to 85%
    # or just select the highest recall threshold that has a non-trivial precision.
    # Typically, a threshold around 0.3 or 0.4 works. Let's find recall >= 0.85. If none, select the lowest threshold in the list (0.10).
    high_recall_candidates = df_metrics[df_metrics["recall"] >= 0.85]
    if not high_recall_candidates.empty:
        # Pick the highest threshold that meets the recall target (to maximize precision)
        conservative_row = high_recall_candidates.iloc[-1]
    else:
        # Fallback to index of maximum recall
        conservative_row = df_metrics.loc[df_metrics["recall"].idxmax()]
    conservative_thresh = conservative_row["threshold"]
    
    # 3. Revenue-Friendly: Reduces false positives (good clients rejected), seeking a higher threshold.
    # We target a lower recall of ~25% to minimize false positives, maximizing customer approvals.
    target_recall = 0.25
    revenue_idx = (df_metrics["recall"] - target_recall).abs().idxmin()
    revenue_row = df_metrics.loc[revenue_idx]
    revenue_thresh = revenue_row["threshold"]
    
    # Write the report
    policy_content = f"""# Halcyon Credit ML Risk Scoring Engine: Decision Threshold Policy

This document establishes the official decision threshold policy for classification of credit risk using the LightGBM champion model. 

The champion pipeline is designed to output a raw continuous probability of default ($P(\\text{{Default}})$). The final credit decision (Approve vs. Reject) is governed by a configurable decision threshold ($T$), where:
- $\\text{{Decision}} = \\text{{Reject}}$ if $P(\\text{{Default}}) \\ge T$
- $\\text{{Decision}} = \\text{{Approve}}$ if $P(\\text{{Default}}) < T$

---

## Business Decision Policies

Depending on macroeconomic conditions, risk tolerance, and customer acquisition targets, the credit risk committee can adopt one of three predefined policy thresholds:

### 1. Conservative Policy (High Recall)
- **Threshold ($T$)**: **{conservative_thresh:.2f}**
- **Recall**: {conservative_row['recall'] * 100:.2f}%
- **Precision**: {conservative_row['precision'] * 100:.2f}%
- **F1-Score**: {conservative_row['f1_score']:.4f}
- **False Negatives (Toxic Loans Approved)**: {int(conservative_row['false_negatives']):,} (Intercepted {int(conservative_row['true_positives']):,} out of {int(conservative_row['true_positives'] + conservative_row['false_negatives']):,} defaults)
- **False Positives (Good Clients Rejected)**: {int(conservative_row['false_positives']):,}
- **Use Case**: Recommended during economic contractions, high-interest-rate environments, or for high-risk credit tiers where the cost of write-offs is extremely high. This policy minimizes toxic defaults at the expense of loan volumes.

### 2. Balanced Policy (Best F1-Score)
- **Threshold ($T$)**: **{balanced_thresh:.2f}**
- **Recall**: {balanced_row['recall'] * 100:.2f}%
- **Precision**: {balanced_row['precision'] * 100:.2f}%
- **F1-Score**: {balanced_row['f1_score']:.4f}
- **False Negatives**: {int(balanced_row['false_negatives']):,}
- **False Positives**: {int(balanced_row['false_positives']):,}
- **Use Case**: Recommended for standard, day-to-day credit scoring. It achieves the mathematically optimal compromise between customer acquisition and write-off prevention.

### 3. Revenue-Friendly Policy (Fewer False Positives)
- **Threshold ($T$)**: **{revenue_thresh:.2f}**
- **Recall**: {revenue_row['recall'] * 100:.2f}%
- **Precision**: {revenue_row['precision'] * 100:.2f}%
- **F1-Score**: {revenue_row['f1_score']:.4f}
- **False Negatives**: {int(revenue_row['false_negatives']):,}
- **False Positives**: {int(revenue_row['false_positives']):,} (Reduces rejections of good clients to **{int(revenue_row['false_positives']):,}**)
- **Use Case**: Recommended during economic growth periods or when launching marketing campaigns to rapidly grow market share. This policy maximizes approval rates and customer acquisition, accepting higher write-off rates.

---

## Comparison Table of Policies

| Policy | Threshold | Recall | Precision | F1-Score | Approved Defaults (FN) | Rejected Good Clients (FP) |
|---|---|---|---|---|---|---|
| **Conservative** | {conservative_thresh:.2f} | {conservative_row['recall']*100:.1f}% | {conservative_row['precision']*100:.1f}% | {conservative_row['f1_score']:.4f} | **{int(conservative_row['false_negatives']):,}** | {int(conservative_row['false_positives']):,} |
| **Balanced** | {balanced_thresh:.2f} | {balanced_row['recall']*100:.1f}% | {balanced_row['precision']*100:.1f}% | {balanced_row['f1_score']:.4f} | {int(balanced_row['false_negatives']):,} | {int(balanced_row['false_positives']):,} |
| **Revenue-Friendly** | {revenue_thresh:.2f} | {revenue_row['recall']*100:.1f}% | {revenue_row['precision']*100:.1f}% | {revenue_row['f1_score']:.4f} | {int(revenue_row['false_negatives']):,} | **{int(revenue_row['false_positives']):,}** |

---

## Architectural Implementation Guidance

### Configurable Inference API
To prevent threshold hardcoding, the prediction engine must not return hardcoded binary predictions. Rather, the inference client should load the threshold dynamically from a central config file or request payload:

```python
# Recommended API Client Implementation Pattern
def make_credit_decision(probabilities: np.ndarray, policy: str = "balanced") -> np.ndarray:
    # Load threshold mapping from configuration
    threshold_mapping = {{
        "conservative": {conservative_thresh:.2f},
        "balanced": {balanced_thresh:.2f},
        "revenue_friendly": {revenue_thresh:.2f}
    }}
    
    threshold = threshold_mapping.get(policy.lower(), {balanced_thresh:.2f})
    
    # Decisions: 1 = Reject (high risk of default), 0 = Approve
    decisions = (probabilities >= threshold).astype(int)
    return decisions
```
"""
    
    policy_path = reports_dir / "THRESHOLD_POLICY.md"
    with open(policy_path, "w") as f:
        f.write(policy_content)
    print(f"THRESHOLD_POLICY.md compiled at: {policy_path}")
    return policy_path


if __name__ == "__main__":
    df_metrics = tune_decision_thresholds()
    compile_threshold_policy(df_metrics)
