"""
Hyperparameter tuning module using GridSearchCV and MLflow for the LightGBM credit risk model.

Exhaustively searches a defined parameter grid to find the optimal ROC-AUC model.
Uses MLflow autologging to track hyperparameter combinations, training runs, and scores.
Refits the best parameters on the full training set for maximum production metrics.
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
from sklearn.model_selection import GridSearchCV, train_test_split
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


def run_grid_search() -> None:
    """
    Exhaustively search hyperparameters using GridSearchCV, log with MLflow,
    evaluate the final champion on the full training set, and deploy if score improves.
    """
    print("Starting hyperparameter tuning run using GridSearchCV & MLflow...")
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
    
    # Subsample training data for the search grid to prevent timeouts (125 combinations * 3 CV = 375 fits)
    print("Preparing 20,000-sample representative subset for hyperparameter search...")
    X_train_search, _, y_train_search, _ = train_test_split(
        X_train, y_train,
        train_size=min(20000, len(X_train)),
        random_state=config.RANDOM_STATE,
        stratify=y_train
    )
    
    # 2. Build baseline pipeline
    preprocessor = get_preprocessor(config)
    lgb_model = LGBMClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        subsample=0.8,
        colsample_bytree=0.8
    )
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", lgb_model)
    ])
    
    # 3. Define target GridSearchCV grid (125 combinations)
    param_grid = {
        "classifier__n_estimators": [50, 100, 150, 200, 250],
        "classifier__max_depth": [4, 6, 8, 10, 12],
        "classifier__learning_rate": [0.001, 0.0025, 0.01, 0.05, 0.1]
    }
    
    # 4. Integrate MLflow Tracking
    import mlflow
    import mlflow.sklearn
    
    mlflow.set_experiment("Credit Risk LightGBM Tuning")
    # Enable autologging for scikit-learn estimators to record cv results
    mlflow.sklearn.autolog(log_models=False)
    
    print(f"Initializing GridSearchCV (125 total combinations, 3-fold CV)...")
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1
    )
    
    # Run search within MLflow context
    print("Fitting GridSearchCV using subset data (375 fits total)...")
    with mlflow.start_run(run_name="GridSearchCV_LightGBM_Search") as run:
        search.fit(X_train_search, y_train_search)
        
        print("\nBest Hyperparameters Found (GridSearchCV):")
        best_params = search.best_params_
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print(f"Best CV ROC-AUC: {search.best_score_:.4f}\n")
        
        # Log CV score to MLflow run
        mlflow.log_metric("best_cv_roc_auc", float(search.best_score_))
        
    # 5. Refit model with optimal parameters on the FULL training set
    print("Refitting pipeline with optimal parameters on the FULL training set for production...")
    # Extract optimal hyperparameter arguments (strip pipeline prefix)
    best_classifier_args = {}
    for param_name, param_val in best_params.items():
        if param_name.startswith("classifier__"):
            best_classifier_args[param_name.split("classifier__")[1]] = param_val
            
    final_lgb_model = LGBMClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        **best_classifier_args
    )
    
    final_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", final_lgb_model)
    ])
    
    # Log the final refit run as a separate production deployment run in MLflow
    with mlflow.start_run(run_name="LGBM_Production_Champion_Refit") as prod_run:
        # Log parameters
        for k, v in best_params.items():
            mlflow.log_param(k, v)
            
        print("Fitting final pipeline on full training data...")
        final_pipeline.fit(X_train, y_train)
        
        # Evaluate best model on test set
        print("Evaluating optimized model on test set...")
        y_pred = final_pipeline.predict(X_test)
        y_prob = final_pipeline.predict_proba(X_test)[:, 1]
        
        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_test, y_prob))
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Log final metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("test_roc_auc", roc_auc)
        
        # Log the scikit-learn model artifact to MLflow using standard pickle
        mlflow.sklearn.log_model(final_pipeline, artifact_path="model", serialization_format="pickle")
        
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
            "model_version": "v1_grid_tuned",
            "training_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save grid tuned metrics
        metrics_path = reports_dir / "lightgbm_grid_tuned_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(tuned_metrics, f, indent=4)
        print(f"Grid tuned metrics saved to: {metrics_path}")
        
        # Save grid tuned model binary
        model_save_path = model_save_dir / "lightgbm_grid_tuned_v1.pkl"
        joblib.dump(final_pipeline, model_save_path)
        print(f"Grid tuned model pipeline saved to: {model_save_path}")
        
        # 6. Compare with prior production model and deploy if better
        prod_metadata_path = prod_dir / "model_metadata.json"
        deploy_new = True
        
        if prod_metadata_path.exists():
            try:
                with open(prod_metadata_path, "r") as f:
                    prod_meta = json.load(f)
                    current_roc = prod_meta.get("ROC-AUC", 0.0)
                    print(f"Current production ROC-AUC: {current_roc:.4f}")
                    print(f"New grid-tuned ROC-AUC: {roc_auc:.4f}")
                    if roc_auc > current_roc:
                        print("New grid-tuned model out-performs the current production champion!")
                    else:
                        print("New grid-tuned model does not out-perform current production champion. Skipping deployment.")
                        deploy_new = False
            except Exception as e:
                print(f"Error loading production metadata: {e}")
                
        if deploy_new:
            # Copy to production
            shutil_dest = prod_dir / "risk_model_v1.pkl"
            shutil.copy2(model_save_path, shutil_dest)
            print(f"Grid-tuned model binary deployed to production champion path: {shutil_dest}")
            
            # Write production metadata
            metadata = {
                "selected_model_name": "LightGBM (Grid Tuned)",
                "selected_model_version": "v1_grid_tuned",
                "reason_for_selection": f"LightGBM tuned via GridSearchCV and tracked with MLflow outperforming prior champion.",
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
            
    print("Grid search tuning and MLflow logging completed successfully.")


if __name__ == "__main__":
    run_grid_search()
