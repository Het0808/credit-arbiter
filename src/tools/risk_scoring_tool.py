"""
Risk Scoring Tool wrapper for the Halcyon Credit Arbiter.

Exposes a clean interface to run model inference on a given application ID
and attach decision support recommendations based on risk band classifications.
"""

from typing import Dict, Any
from src.risk_model.predict import run_applicant_inference


def score_applicant_risk(application_id: int) -> Dict[str, Any]:
    """
    Run risk model inference for a specific client application and attach
    underwriting decision support recommendations.

    Args:
        application_id: Unique client application ID (SK_ID_CURR).

    Returns:
        Dictionary containing scores, risk bands, risk drivers, decision support,
        and model version.
    """
    # 1. Run applicant inference using predict.py
    # This will load the active production pipeline
    prediction = run_applicant_inference(application_id)
    
    risk_band = prediction["risk_band"]
    
    # 2. Map risk band to decision support guidance
    if risk_band == "LOW":
        decision_support = "Eligible for approval if policy checks pass"
    elif risk_band == "MEDIUM":
        decision_support = "Refer to human underwriter"
    elif risk_band == "HIGH":
        decision_support = "High risk; refer to human underwriter with decline consideration"
    else:
        decision_support = "Refer to human underwriter due to unknown risk profile"
        
    # 3. Format result payload
    result = {
        "application_id": prediction["application_id"],
        "risk_score": prediction["risk_score"],
        "risk_band": risk_band,
        "top_risk_factors": prediction["top_risk_factors"],
        "decision_support": decision_support,
        "model_version": prediction["model_version"]
    }
    
    return result


if __name__ == "__main__":
    # Small test run
    import json
    from src.risk_model.config import ModelConfig
    from src.risk_model.preprocess import load_raw_data
    
    config = ModelConfig()
    try:
        df = load_raw_data()
        sample_id = int(df[config.ID_COLUMN].iloc[0])
        print(f"Scoring sample applicant ID: {sample_id}")
        score_res = score_applicant_risk(sample_id)
        print(json.dumps(score_res, indent=2))
    except Exception as e:
        print(f"Error during tool test execution: {e}")
