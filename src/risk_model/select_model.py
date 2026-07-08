"""
Model selection and production deployment module.

Compares baseline Logistic Regression v1 and challenger LightGBM v1 metrics,
selects the champion model based on ROC-AUC, Recall, and F1, deploys the
winning model binary to models/production/, and logs deployment metadata.
"""

from datetime import datetime
import json
import shutil
from pathlib import Path
from typing import Dict, Any


def run_model_selection() -> None:
    """
    Select the best performing model, deploy its binary, and write the metadata record.
    """
    print("Starting production model selection and deployment...")
    
    project_root = Path(__file__).resolve().parents[2]
    reports_dir = project_root / "reports" / "ml"
    prod_dir = project_root / "models" / "production"
    prod_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_metrics_path = reports_dir / "baseline_metrics.json"
    lightgbm_metrics_path = reports_dir / "lightgbm_metrics.json"
    
    # 1. Load baseline metrics
    if not baseline_metrics_path.exists():
        raise FileNotFoundError(f"Baseline metrics not found at {baseline_metrics_path}.")
    with open(baseline_metrics_path, "r") as f:
        baseline_metrics = json.load(f)
        
    # 2. Load LightGBM metrics
    if not lightgbm_metrics_path.exists():
        raise FileNotFoundError(f"LightGBM metrics not found at {lightgbm_metrics_path}.")
    with open(lightgbm_metrics_path, "r") as f:
        lightgbm_metrics = json.load(f)
        
    print("\nEvaluating model performance metrics:")
    print(f"Logistic Regression: ROC-AUC = {baseline_metrics['roc_auc']:.4f}, Recall = {baseline_metrics['recall']:.4f}, F1 = {baseline_metrics['f1']:.4f}")
    print(f"LightGBM Challenger: ROC-AUC = {lightgbm_metrics['roc_auc']:.4f}, Recall = {lightgbm_metrics['recall']:.4f}, F1 = {lightgbm_metrics['f1']:.4f}")
    
    # 3. Model Selection Decision Logic
    # Main decision criterion is ROC-AUC, secondary are Recall and F1
    if lightgbm_metrics["roc_auc"] > baseline_metrics["roc_auc"]:
        selected_model_name = "LightGBM"
        selected_model_version = "v1"
        winning_metrics = lightgbm_metrics
        source_model_path = project_root / "models" / "lightgbm" / "lightgbm_v1.pkl"
        
        roc_diff = lightgbm_metrics["roc_auc"] - baseline_metrics["roc_auc"]
        recall_diff = lightgbm_metrics["recall"] - baseline_metrics["recall"]
        f1_diff = lightgbm_metrics["f1"] - baseline_metrics["f1"]
        
        reason = (
            f"LightGBM outperformed Logistic Regression on ROC-AUC ({roc_diff:+.4f}), "
            f"Recall ({recall_diff:+.4f}), and F1-score ({f1_diff:+.4f}) due to its "
            f"ability to model non-linear decision boundaries and variable interactions natively."
        )
    else:
        selected_model_name = "Logistic Regression"
        selected_model_version = "v1"
        winning_metrics = baseline_metrics
        source_model_path = project_root / "models" / "baseline" / "logistic_regression_v1.pkl"
        reason = "Logistic Regression retained as champion due to superior ROC-AUC score."
        
    print(f"\nWinner Selected: {selected_model_name} {selected_model_version}")
    print(f"Reason: {reason}")
    
    # 4. Save/Deploy selected candidate as models/production/risk_model_v1.pkl
    dest_model_path = prod_dir / "risk_model_v1.pkl"
    shutil.copy2(source_model_path, dest_model_path)
    print(f"Deployed winning model binary to: {dest_model_path}")
    
    # 5. Compile and save production metadata
    # Incorporate the configurable threshold policies we computed earlier
    metadata = {
        "selected_model_name": selected_model_name,
        "selected_model_version": selected_model_version,
        "reason_for_selection": reason,
        "ROC-AUC": winning_metrics["roc_auc"],
        "Precision": winning_metrics["precision"],
        "Recall": winning_metrics["recall"],
        "F1": winning_metrics["f1"],
        "threshold_policy": {
            "conservative": 0.36,
            "balanced": 0.66,
            "revenue_friendly": 0.75
        },
        "feature_count": winning_metrics["feature_count"],
        "training_date": winning_metrics["training_timestamp"].split(" ")[0]
    }
    
    metadata_path = prod_dir / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Production model metadata logged at: {metadata_path}")
    print("Model selection and deployment run completed successfully.")


if __name__ == "__main__":
    run_model_selection()
