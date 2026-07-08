"""
Prediction module for the ML Risk Scoring model.

Exposes functions to load the production model pipeline, run risk score predictions
on applicant details, classify into risk bands, and return explanation factors.
"""

import json
import joblib
from pathlib import Path
from typing import Union, List, Dict, Any
import numpy as np
import pandas as pd
from src.risk_model.config import ModelConfig
from src.risk_model.preprocess import (
    load_raw_data,
    prepare_pipeline_data,
)
from src.risk_model.shap_explain import get_attribution_explanation


# Default path variables
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROD_MODEL_PATH = PROJECT_ROOT / "models" / "production" / "risk_model_v1.pkl"
PROD_METADATA_PATH = PROJECT_ROOT / "models" / "production" / "model_metadata.json"


def load_model(model_path: str = None) -> Any:
    """
    Load the production serialized pipeline.

    Args:
        model_path: Path to serialized model file. Defaults to PROD_MODEL_PATH.

    Returns:
        The loaded estimator or pipeline object.
    """
    if model_path is None:
        model_path = PROD_MODEL_PATH
    else:
        model_path = Path(model_path)
        
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}.")
        
    return joblib.load(model_path)


def predict_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Predict the probability of default for a set of input observations.

    Args:
        model: Loaded model pipeline.
        X: Feature matrix.

    Returns:
        1D array containing probability scores of default.
    """
    return model.predict_proba(X)[:, 1]


def predict_class(model: Any, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    """
    Predict binary credit risk classification (1 for default, 0 for non-default).

    Args:
        model: Loaded model pipeline.
        X: Feature matrix.
        threshold: Classification threshold for mapping probability to class.

    Returns:
        1D array of binary predictions.
    """
    probs = predict_probability(model, X)
    return (probs >= threshold).astype(int)


def calculate_credit_score(
    probability_of_default: float,
    min_score: int = 300,
    max_score: int = 850
) -> int:
    """
    Convert a probability of default into a traditional credit score (300-850 range).

    Args:
        probability_of_default: The probability of default (0.0 to 1.0).
        min_score: Lower bound of the credit score range (e.g., FICO minimum 300).
        max_score: Upper bound of the credit score range (e.g., FICO maximum 850).

    Returns:
        An integer credit score.
    """
    score = max_score - (probability_of_default * (max_score - min_score))
    return int(np.round(score))


def calculate_risk_band(
    probability: float,
    low_threshold: float = 0.36,
    high_threshold: float = 0.66
) -> str:
    """
    Categorize risk probability into LOW, MEDIUM, or HIGH risk bands.

    Args:
        probability: Probability of default (0.0 to 1.0).
        low_threshold: Upper limit for LOW risk.
        high_threshold: Lower limit for HIGH risk.

    Returns:
        Risk band string ('LOW', 'MEDIUM', 'HIGH').
    """
    if probability < low_threshold:
        return "LOW"
    elif probability < high_threshold:
        return "MEDIUM"
    else:
        return "HIGH"


def run_applicant_inference(
    application_id: int,
    low_threshold: float = 0.36,
    high_threshold: float = 0.66
) -> Dict[str, Any]:
    """
    Load an applicant's data, run inference, calculate risk band,
    and attach explanation factors.

    Args:
        application_id: Unique client application ID (SK_ID_CURR).
        low_threshold: Decision boundary for LOW risk band.
        high_threshold: Decision boundary for HIGH risk band.

    Returns:
        Dictionary containing prediction results, scores, bands, risk factors,
        model version, and type.
    """
    config = ModelConfig()
    
    # 1. Load production model
    model = load_model(PROD_MODEL_PATH)
    
    # Load model metadata
    model_version = "v1"
    model_type = "LightGBM"
    if PROD_METADATA_PATH.exists():
        try:
            with open(PROD_METADATA_PATH, "r") as f:
                metadata = json.load(f)
                model_version = metadata.get("selected_model_version", model_version)
                model_type = metadata.get("selected_model_name", model_type)
        except Exception as e:
            print(f"Warning: Could not read metadata: {e}")
            
    # 2. Load raw dataset and extract target row
    df = load_raw_data()
    applicant_df = df[df[config.ID_COLUMN] == application_id]
    
    if applicant_df.empty:
        raise ValueError(f"Application ID {application_id} not found in raw dataset.")
        
    # 3. Apply feature engineering
    X, _, _ = prepare_pipeline_data(applicant_df, config)
    
    # 4. Predict probability
    prob = float(predict_probability(model, X)[0])
    
    # 5. Classify risk band
    risk_band = calculate_risk_band(prob, low_threshold, high_threshold)
    
    # 6. Retrieve explanation factors
    explanation = get_attribution_explanation(application_id)
    top_factors = explanation.get("top_risk_drivers", [])
    
    # Format return payload
    result = {
        "application_id": int(application_id),
        "risk_score": float(np.round(prob, 4)),
        "risk_band": risk_band,
        "top_risk_factors": top_factors,
        "model_version": model_version,
        "model_type": model_type
    }
    
    return result


if __name__ == "__main__":
    # Test sample inference run
    config = ModelConfig()
    try:
        df = load_raw_data()
        sample_id = int(df[config.ID_COLUMN].iloc[0])
        print(f"Running inference on sample applicant: {sample_id}")
        result = run_applicant_inference(sample_id)
        print("\nInference Output JSON:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error during test inference: {e}")
