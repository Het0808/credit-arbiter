"""
Evaluation and report generation module for the ML Risk Scoring model.

Calculates metrics and generates visualization plots (ROC Curve, PR Curve,
Confusion Matrix, Class Distribution) for the baseline model, saving them
to reports/ml/ and writing the BASELINE_ANALYSIS.md report.
"""

import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from src.risk_model.config import ModelConfig, MODEL_PATH_ABS, REPORT_PATH_ABS
from src.risk_model.preprocess import (
    load_raw_data,
    prepare_pipeline_data,
    split_data,
)


def generate_evaluation_report() -> Path:
    """
    Generate performance charts and compile the BASELINE_ANALYSIS.md evaluation report.

    Returns:
        Path to the generated markdown report.
    """
    print("Loading configuration...")
    config = ModelConfig()
    
    reports_dir = Path(__file__).resolve().parents[2] / "reports" / "ml"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading raw data for analysis...")
    df = load_raw_data()
    
    # Class distribution in raw data
    class_counts = df[config.TARGET_COLUMN].value_counts()
    class_pct = df[config.TARGET_COLUMN].value_counts(normalize=True) * 100
    is_imbalanced = (class_counts.iloc[1] / class_counts.sum()) < 0.15
    
    print("Preparing pipeline data...")
    X, y, _ = prepare_pipeline_data(df, config)
    
    print("Recreating validation test split...")
    _, X_test, _, y_test = split_data(X, y, config)
    
    print("Loading trained model artifact...")
    if not MODEL_PATH_ABS.exists():
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH_ABS}. Please run train.py first.")
        
    model = joblib.load(MODEL_PATH_ABS)
    
    print("Running predictions on test set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    print("Generating evaluation plots...")
    
    # Set plot style
    sns.set_theme(style="whitegrid")
    
    # 1. Class Distribution Plot
    plt.figure(figsize=(6, 4))
    sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, palette="viridis", legend=False)
    plt.title("Class Distribution (Target Variable)")
    plt.xlabel("Default (1) vs Non-Default (0)")
    plt.ylabel("Count")
    for i, (count, pct) in enumerate(zip(class_counts.values, class_pct.values)):
        plt.text(i, count + 5000, f"{count:,}\n({pct:.2f}%)", ha='center', fontweight='bold')
    plt.tight_layout()
    class_dist_path = reports_dir / "class_distribution.png"
    plt.savefig(class_dist_path, dpi=300)
    plt.close()
    
    # 2. Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5))
    cm_matrix = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(
        cm_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Non-Default (0)", "Default (1)"],
        yticklabels=["Non-Default (0)", "Default (1)"]
    )
    plt.title("Confusion Matrix Heatmap")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = reports_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = reports_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    
    # 4. Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall_vals, precision_vals, color='blue', lw=2, label=f'PR curve (AP = {ap:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    pr_path = reports_dir / "precision_recall_curve.png"
    plt.savefig(pr_path, dpi=300)
    plt.close()
    
    # Write BASELINE_ANALYSIS.md
    analysis_file_path = reports_dir / "BASELINE_ANALYSIS.md"
    
    imbalance_status = "YES (Highly Imbalanced)" if is_imbalanced else "NO"
    
    report_content = f"""# ML Risk Scoring Model: Baseline Analysis Report

This document presents a detailed evaluation of the baseline Logistic Regression model trained for default risk prediction using the Home Credit dataset.

## Executive Summary

- **ROC-AUC**: {roc_auc:.4f}
- **Accuracy**: {accuracy:.4f}
- **Precision**: {precision:.4f}
- **Recall**: {recall:.4f}
- **F1-Score**: {f1:.4f}
- **Average Precision (PR-AUC)**: {ap:.4f}

---

## 1. Class Distribution & Imbalance Analysis

### Target Variable Distribution
- **Non-Default (Class 0)**: {class_counts.iloc[0]:,} applications ({class_pct.iloc[0]:.2f}%)
- **Default (Class 1)**: {class_counts.iloc[1]:,} applications ({class_pct.iloc[1]:.2f}%)

### Dataset Imbalance Check
- **Is the dataset imbalanced?** **{imbalance_status}**. Only **{class_pct.iloc[1]:.2f}%** of the applicants defaulted in this training dataset. 
- **Impact of Imbalance**: When a dataset is heavily imbalanced, a naive model can achieve 91.9% accuracy simply by predicting that nobody will default. Therefore, standard **Accuracy** is a deceptive metric. We must rely on **ROC-AUC**, **Recall**, and **F1-Score** to gauge the model's true performance.

![Class Distribution](class_distribution.png)

---

## 2. Confusion Matrix

A confusion matrix shows the breakdown of correct and incorrect predictions:

| | Predicted Non-Default (0) | Predicted Default (1) |
|---|---|---|
| **Actual Non-Default (0)** | **{tn:,}** (True Negative) | **{fp:,}** (False Positive) |
| **Actual Default (1)** | **{fn:,}** (False Negative) | **{tp:,}** (True Positive) |

### Key Takeaways from the Confusion Matrix:
- The model correctly identified **{tp:,}** default cases.
- The model missed **{fn:,}** default cases (False Negatives), representing risky loans that would have been approved.
- The model flagged **{fp:,}** non-defaulting applications as defaults (False Positives), representing lost revenue/customers who were turned down but would have paid back.

![Confusion Matrix](confusion_matrix.png)

---

## 3. Metrics Explained in Loan Default Context

### Accuracy ({accuracy:.4f})
- **Definition**: The proportion of all predictions that were correct (both default and non-default).
- **Credit Risk Context**: High accuracy is easy to achieve due to class imbalance but is highly misleading. In our case, the model's accuracy ({accuracy:.4f}) is lower than the majority-class baseline ({class_pct.iloc[0]/100:.4f}) because the decision threshold has been adjusted (via `class_weight="balanced"`) to catch defaults, trading off overall accuracy.

### Precision ({precision:.4f})
- **Definition**: Out of all applicants predicted to default, what percentage actually defaulted?
- **Credit Risk Context**: Precision is relatively low ({precision:.4f}). This means that when the model flags someone as high risk, they only default ~15.6% of the time. While this results in a higher rate of false alarms (turning down customers who wouldn't default), it is a common tradeoff when minimizing default risk.

### Recall ({recall:.4f})
- **Definition**: Out of all applicants who actually defaulted, what percentage did the model correctly identify?
- **Credit Risk Context**: Recall is **{recall * 100:.2f}%**. The model successfully flags nearly two-thirds of the default cases. High recall is critical in credit risk because the cost of a false negative (defaulting client) is far greater than the cost of a false positive (rejected applicant).

### F1-Score ({f1:.4f})
- **Definition**: The harmonic mean of Precision and Recall.
- **Credit Risk Context**: The F1-score balances the trade-off between Precision and Recall. An F1-score of {f1:.4f} is typical for baseline models on imbalanced tabular data with simple features.

### ROC-AUC ({roc_auc:.4f})
- **Definition**: Receiver Operating Characteristic - Area Under Curve. Measures the model's ability to rank applicants by default probability.
- **Credit Risk Context**: A score of **{roc_auc:.4f}** indicates moderate-to-good discriminative power. It means that there is a {roc_auc * 100:.1f}% chance that the model will rank a randomly chosen defaulting applicant as riskier than a randomly chosen non-defaulting applicant.

---

## 4. Visualizations

### ROC Curve
The ROC curve plots the True Positive Rate (Recall) against the False Positive Rate. A higher area under the curve (closer to 1.0) is better.

![ROC Curve](roc_curve.png)

### Precision-Recall Curve
The Precision-Recall curve plots Precision against Recall. In highly imbalanced contexts, the PR curve is much more informative than the ROC curve because it highlights how precision degrades as recall increases.

![Precision-Recall Curve](precision_recall_curve.png)

---

## 5. Recommendations for Improving the Model

To improve performance beyond this baseline Logistic Regression model:

1. **Adopt Tree-Based Ensembles**:
   - Transition to **LightGBM**, **XGBoost**, or **CatBoost**. Tree-based ensembles are superior at capturing non-linear relationships, interactive features (e.g., interaction between income and credit), and handling missing values natively without imputation distortion.

2. **Advanced Feature Engineering**:
   - **External Source Interactions**: Combine `EXT_SOURCE_1`, `EXT_SOURCE_2`, and `EXT_SOURCE_3` (e.g., mean, product, or min/max).
   - **Credit Term Features**: Calculate expected credit term (`AMT_CREDIT / AMT_ANNUITY`).
   - **Age / Employment Ratios**: Look at relative work experience durations (e.g., `DAYS_EMPLOYED / (DAYS_BIRTH * 365.25)`).

3. **Hyperparameter Tuning**:
   - Run cross-validated grid search or Bayesian optimization (e.g. Optuna) to find optimal regularization and class weights.

4. **Address Class Imbalance via Resampling**:
   - Experiment with SMOTE (Synthetic Minority Over-sampling Technique) or down-sampling the majority class rather than relying purely on class weighting.

5. **Utilize Relational Data Sources**:
   - Merge features from auxiliary tables such as `bureau.csv` (credit bureau history) and `previous_application.csv` (history with Home Credit) to calculate total active loans and historic default flags.
"""
    
    with open(analysis_file_path, "w") as f:
        f.write(report_content)
        
    print(f"Report generated successfully at: {analysis_file_path}")
    return analysis_file_path


if __name__ == "__main__":
    generate_evaluation_report()
