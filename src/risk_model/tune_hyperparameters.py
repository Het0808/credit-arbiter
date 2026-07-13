"""
Hyperparameter tuning module for the LightGBM credit risk model.

Performs randomized cross-validation search (RandomizedSearchCV) over LightGBM
hyperparameters to optimize ROC-AUC, evaluates results, and optionally deploys
the tuned champion model to production.
"""

from datetime import datetime
import json
import joblib
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from src.risk_model.config import ModelConfig
from src.risk_model.preprocess import (
    load_raw_data,
    prepare_pipeline_data,
    split_data,
    get_preprocessor,
)


def tune_and_evaluate() -> None:
    """
    Search hyperparameters, evaluate, and deploy tuned LightGBM model.
    """
    print("Starting hyperparameter tuning run for LightGBM...")
    config = ModelConfig()
    
    # Paths
    project_root = Path(__file__).resolve().parents[2]
    model_save_dir = project_root / "models" / "lightgbm"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    prod_dir = project_root / "models" / "production"
    prod_dir.mkdir(parents=True, exist_ok=True)
    
    reports_dir = project_root / "reports" / "ml"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    print("Loading data...")
    df = load_raw_data()
    X, y, _ = prepare_pipeline_data(df, config)
    X_train, X_test, y_train, y_test = split_data(X, y, config)
    
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = float(neg_count / pos_count)
    
    # 2. Build baseline pipeline
    preprocessor = get_preprocessor(config)
    lgb_model = LGBMClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", lgb_model)
    ])
    
    # 3. Define hyperparameter search distribution
    # Keep search space reasonably compact for rapid execution
    param_dist = {
        "classifier__n_estimators": [100, 150, 200],
        "classifier__max_depth": [4, 6, 8],
        "classifier__num_leaves": [15, 31, 63],
        "classifier__learning_rate": [0.03, 0.05, 0.1],
        "classifier__subsample": [0.7, 0.8, 0.9],
        "classifier__colsample_bytree": [0.7, 0.8, 0.9]
    }
    
    print("Initializing RandomizedSearchCV (8 iterations, 3-fold CV)...")
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=8,
        cv=3,
        scoring="roc_auc",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    
    # Run search
    print("Fitting RandomizedSearchCV on training set (this may take a minute)...")
    search.fit(X_train, y_train)
    
    print("\nBest Hyperparameters Found:")
    best_params = search.best_params_
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}\n")
    
    # Evaluate best model
    best_pipeline = search.best_estimator_
    print("Evaluating optimized model on test set...")
    y_pred = best_pipeline.predict(X_test)
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]
    
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    tuned_metrics = {
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
        "best_params": best_params,
        "cv_roc_auc": float(search.best_score_),
        "feature_count": int(X_train.shape[1]),
        "model_version": "v1_tuned",
        "training_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save tuned metrics
    metrics_path = reports_dir / "lightgbm_tuned_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(tuned_metrics, f, indent=4)
    print(f"Tuned metrics saved to: {metrics_path}")
    
    # Save tuned model binary
    model_save_path = model_save_dir / "lightgbm_tuned_v1.pkl"
    joblib.dump(best_pipeline, model_save_path)
    print(f"Tuned model pipeline saved to: {model_save_path}")
    
    # 4. Compare with prior production model and deploy if better
    prod_metadata_path = prod_dir / "model_metadata.json"
    deploy_new = True
    
    if prod_metadata_path.exists():
        try:
            with open(prod_metadata_path, "r") as f:
                prod_meta = json.load(f)
                current_roc = prod_meta.get("ROC-AUC", 0.0)
                print(f"Current production ROC-AUC: {current_roc:.4f}")
                print(f"New tuned ROC-AUC: {roc_auc:.4f}")
                if roc_auc > current_roc:
                    print("New tuned model out-performs the current production champion!")
                else:
                    print("New tuned model does not out-perform current production champion. Skipping deployment.")
                    deploy_new = False
        except Exception as e:
            print(f"Error loading production metadata: {e}")
            
    if deploy_new:
        # Copy to production
        shutil_dest = prod_dir / "risk_model_v1.pkl"
        shutil.copy2(model_save_path, shutil_dest)
        print(f"Tuned model binary deployed to production champion path: {shutil_dest}")
        
        # Write production metadata
        metadata = {
            "selected_model_name": "LightGBM (Tuned)",
            "selected_model_version": "v1_tuned",
            "reason_for_selection": f"LightGBM tuned via RandomizedSearchCV (CV ROC-AUC = {search.best_score_:.4f}) outperforming prior champion.",
            "ROC-AUC": roc_auc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "threshold_policy": {
                "conservative": 0.36,
                "balanced": 0.66,
                "revenue_friendly": 0.75
            },
            "feature_count": int(X_train.shape[1]),
            "training_date": datetime.now().strftime("%Y-%m-%d")
        }
        with open(prod_metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Production model metadata updated at: {prod_metadata_path}")
        
    print("Tuning run completed successfully.")


if __name__ == "__main__":
    tune_and_evaluate()
