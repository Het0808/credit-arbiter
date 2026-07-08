"""
Training module for the ML Risk Scoring model.

Contains the pipeline to train the model, tune hyperparameters, evaluate performance,
and save the resulting model artifacts.
"""

import json
import joblib
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from src.risk_model.config import ModelConfig, MODEL_PATH_ABS, REPORT_PATH_ABS
from src.risk_model.preprocess import (
    load_raw_data,
    prepare_pipeline_data,
    split_data,
    get_preprocessor,
)


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, config: ModelConfig
) -> Pipeline:
    """
    Train a baseline Logistic Regression model based on the configuration parameters.

    Args:
        X_train: Feature matrix for training.
        y_train: Target labels for training.
        config: ModelConfig containing model type and hyperparameters.

    Returns:
        The trained scikit-learn Pipeline containing preprocessor and estimator.
    """
    # Create the ColumnTransformer preprocessor
    preprocessor = get_preprocessor(config)
    
    # Initialize a baseline Logistic Regression model
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=config.RANDOM_STATE
    )
    
    # Build complete sklearn pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    
    # Fit the model pipeline
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, Any]:
    """
    Evaluate the trained model on test data, computing key metrics like ROC AUC,
    accuracy, precision, recall, F1-score, and confusion matrix.

    Args:
        model: Trained model pipeline/estimator.
        X_test: Feature matrix for testing.
        y_test: True target labels for testing.

    Returns:
        Dictionary containing metric names and their corresponding values.
    """
    # Predict classes and probability of default (positive class 1)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Compute standard classification metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    metrics = {
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }
    return metrics


def save_model_artifact(model: Any, model_path: str = None) -> str:
    """
    Persist the trained model pipeline/estimator to the model registry/directory.

    Args:
        model: Trained model pipeline/estimator.
        model_path: Filename path to save the model under. Defaults to MODEL_PATH_ABS.

    Returns:
        The file path where the model was successfully saved.
    """
    if model_path is None:
        model_path = MODEL_PATH_ABS
    else:
        from pathlib import Path
        model_path = Path(model_path)

    # Ensure parent directory exists
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Serialize model pipeline using joblib
    joblib.dump(model, model_path)
    return str(model_path)


def save_metrics(metrics: Dict[str, Any], report_path: str = None) -> str:
    """
    Save evaluation metrics dictionary to a JSON report file.

    Args:
        metrics: Dictionary containing performance metrics.
        report_path: Path to save JSON report. Defaults to REPORT_PATH_ABS.

    Returns:
        The file path where the metrics report was successfully saved.
    """
    if report_path is None:
        report_path = REPORT_PATH_ABS
    else:
        from pathlib import Path
        report_path = Path(report_path)

    # Ensure parent directory exists
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as formatted JSON
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    return str(report_path)


if __name__ == "__main__":
    print("Starting ML risk model training pipeline...")
    
    # 1. Load config
    config = ModelConfig()
    
    # 2. Load dataset
    print("Loading data...")
    df = load_raw_data()
    
    # 3. Preprocess and split
    print("Preprocessing data and constructing features...")
    X, y, _ = prepare_pipeline_data(df, config)
    
    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y, config)
    
    print(f"Dataset stats: Train features shape = {X_train.shape}")
    print(f"Target distribution (class 1 is default):")
    print(y_train.value_counts(normalize=True).to_string())
    
    # 4. Train pipeline
    print("Fitting Logistic Regression baseline pipeline...")
    pipeline = train_model(X_train, y_train, config)
    
    # 5. Evaluate pipeline
    print("Evaluating model performance...")
    metrics = evaluate_model(pipeline, X_test, y_test)
    
    # Print metrics cleanly to stdout
    print("\n===============================")
    print("     BASELINE MODEL METRICS    ")
    print("===============================")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-Score : {metrics['f1']:.4f}")
    print("-------------------------------")
    print("Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"  [TN: {cm['true_negatives']:5d} | FP: {cm['false_positives']:5d}]")
    print(f"  [FN: {cm['false_negatives']:5d} | TP: {cm['true_positives']:5d}]")
    print("===============================\n")
    
    # 6. Save model pipeline
    print("Saving trained model pipeline...")
    saved_model_path = save_model_artifact(pipeline)
    print(f"Artifact persisted at: {saved_model_path}")
    
    # 7. Save performance metrics
    print("Saving metrics report...")
    saved_report_path = save_metrics(metrics)
    print(f"Report saved at: {saved_report_path}")
    
    print("Training run completed successfully.")

