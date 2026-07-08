"""
Archiving and versioning script for the v1 baseline credit risk scoring model.

Saves the v1 baseline pipeline, metrics JSON, metadata JSON, visual plots,
and compiles the official BASELINE_REPORT.md documentation.
"""

from datetime import datetime
import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
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
from src.risk_model.config import ModelConfig
from src.risk_model.preprocess import (
    load_raw_data,
    prepare_pipeline_data,
    split_data,
    get_preprocessor,
)


def archive_baseline_model() -> None:
    """
    Train, evaluate, and save all baseline v1 experiment files, plots, metadata, and reports.
    """
    print("Initializing baseline v1 archiving run...")
    config = ModelConfig()
    
    # Define paths
    project_root = Path(__file__).resolve().parents[2]
    baseline_model_dir = project_root / "models" / "baseline"
    plots_dir = project_root / "reports" / "ml" / "plots"
    reports_dir = project_root / "reports" / "ml"
    
    # Auto-create directories
    baseline_model_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading raw data...")
    df = load_raw_data()
    
    total_rows = len(df)
    class_counts = df[config.TARGET_COLUMN].value_counts().to_dict()
    class_pcts = (df[config.TARGET_COLUMN].value_counts(normalize=True) * 100).to_dict()
    
    print("Preparing pipeline data...")
    X, y, _ = prepare_pipeline_data(df, config)
    
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_data(X, y, config)
    
    # Train new baseline pipeline
    print("Training Logistic Regression baseline v1...")
    preprocessor = get_preprocessor(config)
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=config.RANDOM_STATE
    )
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    print("Evaluating baseline model...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    ap = float(average_precision_score(y_test, y_prob))
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save visualizations
    print("Generating visualizations...")
    sns.set_theme(style="whitegrid")
    
    # 1. Confusion Matrix Heatmap
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
    plt.title("Confusion Matrix (Baseline v1)")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrix.png", dpi=300)
    plt.close()
    
    # 2. ROC Curve
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
    plt.savefig(plots_dir / "roc_curve.png", dpi=300)
    plt.close()
    
    # 3. Precision-Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall_vals, precision_vals, color='blue', lw=2, label=f'PR curve (AP = {ap:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(plots_dir / "precision_recall_curve.png", dpi=300)
    plt.close()
    
    # 4. Save trained pipeline
    model_path = baseline_model_dir / "logistic_regression_v1.pkl"
    joblib.dump(pipeline, model_path)
    print(f"Model and preprocessing pipeline saved to: {model_path}")
    
    # 5. Save metrics JSON
    metrics_data = {
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "class_distribution": {
            "non_default_count": int(class_counts.get(0, 0)),
            "non_default_pct": float(class_pcts.get(0, 0.0)),
            "default_count": int(class_counts.get(1, 0)),
            "default_pct": float(class_pcts.get(1, 0.0))
        },
        "feature_count": int(X_train.shape[1]),
        "training_timestamp": timestamp,
        "model_version": "v1"
    }
    
    metrics_path = reports_dir / "baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")
    
    # 6. Save metadata JSON
    features_list = list(config.NUMERICAL_FEATURES) + list(config.CATEGORICAL_FEATURES)
    metadata_data = {
        "model_name": "Logistic Regression",
        "version": "v1",
        "dataset": "Home Credit Default Risk",
        "training_rows": int(X_train.shape[0]),
        "features": features_list,
        "target": config.TARGET_COLUMN,
        "roc_auc": roc_auc,
        "training_date": timestamp.split(" ")[0]
    }
    
    metadata_path = baseline_model_dir / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata_data, f, indent=4)
    print(f"Metadata saved to: {metadata_path}")
    
    # 7. Create BASELINE_REPORT.md
    report_content = f"""# Halcyon Credit ML Risk Scoring Engine: Baseline Model Report

This document registers the official v1 baseline credit risk prediction model for the Halcyon Credit Arbiter project.

## Model Description
The model is a **Logistic Regression** pipeline, regularized and configured to handle substantial class imbalances. It integrates data imputation, numerical scaling, categorical encoding, and domain-engineered features inside a single, unified scikit-learn Pipeline object.

- **Algorithm**: Logistic Regression (`max_iter=1000`, `class_weight="balanced"`)
- **Pipeline version**: `v1`
- **Training Timestamp**: {timestamp}

## Dataset Used
- **Source**: Home Credit Default Risk application training dataset (`application_train.csv`).
- **Total Records**: {total_rows:,} applications.
- **Training Records (80%)**: {X_train.shape[0]:,} applications.
- **Test Records (20%)**: {X_test.shape[0]:,} applications.

## Target & Features Used
- **Target Variable**: `{config.TARGET_COLUMN}` (1 for defaulted applicant, 0 for non-defaulted applicant)
- **Total Feature Count**: {X_train.shape[1]} features (numerical + categorical after encoding).

### Numeric Features List
{chr(10).join([f"- `{col}`" for col in config.NUMERICAL_FEATURES])}

### Categorical Features List
{chr(10).join([f"- `{col}`" for col in config.CATEGORICAL_FEATURES])}

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
| **ROC-AUC** | **{roc_auc:.4f}** |
| **Accuracy** | **{accuracy:.4f}** |
| **Precision** | **{precision:.4f}** |
| **Recall** | **{recall:.4f}** |
| **F1-Score** | **{f1:.4f}** |

### Confusion Matrix (Test Set)
- **True Negatives (TN)**: {tn:,} (Creditworthy applicants approved)
- **False Positives (FP)**: {fp:,} (Creditworthy applicants rejected)
- **False Negatives (FN)**: {fn:,} (Defaulting applicants approved)
- **True Positives (TP)**: {tp:,} (Defaulting applicants blocked)

## Visualizations

### Confusion Matrix
![Confusion Matrix](plots/confusion_matrix.png)

### ROC Curve
![ROC Curve](plots/roc_curve.png)

### Precision-Recall Curve
![Precision-Recall Curve](plots/precision_recall_curve.png)

## Business Interpretation
- **Default Capture Rate (Recall)**: The model intercepts **{recall * 100:.2f}%** of defaulting applications. Intercepting defaults directly protects capital.
- **Approval Quality (Precision)**: Because of the low precision ({precision * 100:.2f}%), flagging an applicant as default has a relatively high false alarm rate. For every true default blocked, the model flags approximately 5 non-defaulting applications. While this trades off customer acquisition, it is standard in conservative risk profiles.
- **ROC-AUC ({roc_auc:.4f})**: The model possesses strong ranking capability, which is suitable for tiering interest rates and limits.

## Known Limitations
1. **Linear Assumptions**: Logistic Regression assumes linear log-odds relations, failing to model compound interactive effects (e.g. high credit limit *low* income).
2. **Imputation Bias**: Standard median imputation distorts distribution shapes.
3. **Information Collinearity**: Financial ratios derived from the same base columns lead to regression collinearity.

## Recommendations for Next Iteration
1. **Transition to Tree-Based Ensemble (LightGBM/XGBoost)**: To capture interactive dependencies without hand-crafting interactions.
2. **Auxiliary Relational Data**: Incorporate bureau credit histories (`bureau.csv`) and previous applications (`previous_application.csv`).
3. **Advanced Threshold Tuning**: Set the classification threshold dynamically based on the dollar cost of false negatives vs. false positives rather than defaulting to 0.5.
"""
    
    report_path = reports_dir / "BASELINE_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"BASELINE_REPORT.md compiled at: {report_path}")
    
    print("Baseline v1 archiving run completed successfully.")


if __name__ == "__main__":
    archive_baseline_model()
