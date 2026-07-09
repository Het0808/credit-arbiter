"""
Prediction module for the ML Risk Scoring model.

Exposes functions to load the production model pipeline, run risk score predictions
on applicant details, classify into risk bands, and return explanation factors.
"""

import json
import joblib
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from src.risk_model.config import ModelConfig
from src.risk_model.preprocess import clean_data, engineer_features


# Default path variables
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROD_MODEL_PATH = PROJECT_ROOT / "models" / "production" / "risk_model_v1.pkl"
PROD_METADATA_PATH = PROJECT_ROOT / "models" / "production" / "model_metadata.json"

# Human-readable labels for the raw/engineered feature names surfaced in SHAP
# explanations (US-202 AC: "each feature has a human-readable label").
FEATURE_LABELS: Dict[str, str] = {
    "AMT_INCOME_TOTAL": "Annual income",
    "AMT_CREDIT": "Loan amount",
    "AMT_ANNUITY": "Monthly loan payment",
    "DAYS_EMPLOYED": "Employment tenure (days)",
    "EXT_SOURCE_1": "External credit bureau score 1",
    "EXT_SOURCE_2": "External credit bureau score 2",
    "EXT_SOURCE_3": "External credit bureau score 3",
    "CNT_FAM_MEMBERS": "Family size",
    "AMT_GOODS_PRICE": "Price of goods financed",
    "CNT_CHILDREN": "Number of children",
    "CREDIT_INCOME_RATIO": "Loan amount relative to income",
    "ANNUITY_INCOME_RATIO": "Monthly payment relative to income",
    "CREDIT_ANNUITY_RATIO": "Loan term proxy (credit / annuity)",
    "CREDIT_GOODS_RATIO": "Loan-to-value ratio",
    "EMPLOYMENT_YEARS": "Years employed",
    "CHILDREN_RATIO": "Children as share of household",
    "INCOME_PER_PERSON": "Income per household member",
    "EXT_SOURCE_MEAN": "Average external credit bureau score",
    "EXT_SOURCE_STD": "External credit bureau score volatility",
    "EXT_SOURCE_MAX": "Best external credit bureau score",
    "EXT_SOURCE_MIN": "Worst external credit bureau score",
    "TOTAL_MISSING_VALUES": "Missing data points on file",
    "MISSING_PERCENTAGE": "Percentage of profile incomplete",
    "TOTAL_DOCUMENT_FLAGS": "Supporting documents on file",
    "NAME_CONTRACT_TYPE": "Loan contract type",
    "FLAG_OWN_CAR": "Owns a car",
    "FLAG_OWN_REALTY": "Owns real estate",
    "NAME_INCOME_TYPE": "Income source type",
    "NAME_EDUCATION_TYPE": "Education level",
}

# Lazily-populated module-level cache so the production pipeline + SHAP
# explainer are loaded once per process, not once per request (needed to stay
# within the <=300ms ML inference NFR budget).
_CACHE: Dict[str, Any] = {"pipeline": None, "explainer": None, "metadata": None}


def _profile_to_raw_row(profile: Dict[str, Any]) -> pd.DataFrame:
    """Map a normalised applicant profile dict (Application model field names)
    to a single-row DataFrame using the original HC2018 raw column names, so it
    can be run through the same engineer_features() used at training time.

    Deliberately never reads code_gender or days_birth (assumption A-8b) - it
    only maps the keys listed below, so passing those extra keys in `profile`
    (as fairness.py callers may) has no effect on the output.
    """
    numeric_fields = {
        "AMT_INCOME_TOTAL": profile.get("amt_income_total"),
        "AMT_CREDIT": profile.get("amt_credit"),
        "AMT_ANNUITY": profile.get("amt_annuity"),
        "DAYS_EMPLOYED": profile.get("days_employed"),
        "EXT_SOURCE_1": profile.get("ext_source_1"),
        "EXT_SOURCE_2": profile.get("ext_source_2"),
        "EXT_SOURCE_3": profile.get("ext_source_3"),
        "CNT_FAM_MEMBERS": profile.get("cnt_fam_members"),
        "AMT_GOODS_PRICE": profile.get("amt_goods_price"),
        "CNT_CHILDREN": profile.get("cnt_children"),
    }
    categorical_fields = {
        "NAME_CONTRACT_TYPE": profile.get("name_contract_type"),
        "FLAG_OWN_CAR": profile.get("flag_own_car"),
        "FLAG_OWN_REALTY": profile.get("flag_own_realty"),
        "NAME_INCOME_TYPE": profile.get("name_income_type"),
        "NAME_EDUCATION_TYPE": profile.get("name_education_type"),
    }
    df = pd.DataFrame([{**numeric_fields, **categorical_fields}])
    # A single-row frame with a None value infers dtype=object for that
    # column, which breaks the arithmetic in engineer_features() (None isn't
    # treated as NaN by pandas' division operators the way a real NaN is).
    # Coerce the numeric columns explicitly so missing values become proper
    # NaN and are imputed downstream, same as they would be from a real CSV.
    for col in numeric_fields:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _get_production_pipeline() -> Any:
    """Load (and cache) the production model pipeline."""
    if _CACHE["pipeline"] is None:
        _CACHE["pipeline"] = load_model(PROD_MODEL_PATH)
    return _CACHE["pipeline"]


def _get_production_metadata() -> Dict[str, Any]:
    if _CACHE["metadata"] is None:
        metadata = {"selected_model_version": "v1", "selected_model_name": "LightGBM"}
        if PROD_METADATA_PATH.exists():
            try:
                with open(PROD_METADATA_PATH, "r") as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Could not read metadata: {e}")
        _CACHE["metadata"] = metadata
    return _CACHE["metadata"]


def _get_explainer(classifier: Any) -> Any:
    """Load (and cache) a SHAP TreeExplainer for the champion classifier.

    No background dataset is supplied, so SHAP defaults to tree_path_dependent
    perturbation - accurate for tree ensembles and fast enough for per-request
    use (no CSV load, no plot generation), unlike the offline shap_explain.py
    analysis path which is for batch/report use only.
    """
    if _CACHE["explainer"] is None:
        import shap

        _CACHE["explainer"] = shap.TreeExplainer(classifier)
    return _CACHE["explainer"]


def predict_from_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Score a normalised applicant profile directly (no CSV/SK_ID_CURR lookup).

    This is the live-request path used by src/api/services/scoring.py: it
    accepts the same profile shape produced by the ingestion pipeline, runs it
    through the same feature engineering used at training time, and returns a
    probability, risk band, and top-5 SHAP-attributed risk factors with
    human-readable labels.

    Args:
        profile: Normalised applicant profile dict (Application field names).

    Returns:
        Dict with risk_score, risk_band, top_risk_factors, model_version, model_type.
    """
    config = ModelConfig()
    pipeline = _get_production_pipeline()
    metadata = _get_production_metadata()

    raw_df = _profile_to_raw_row(profile)
    engineered = engineer_features(clean_data(raw_df))

    all_features = list(config.NUMERICAL_FEATURES) + list(config.CATEGORICAL_FEATURES)
    X = engineered.reindex(columns=all_features)

    prob = float(predict_probability(pipeline, X)[0])
    risk_band = calculate_risk_band(prob)

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    X_transformed = preprocessor.transform(X)
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

    explainer = _get_explainer(classifier)
    shap_values = explainer(X_transformed_df)
    shap_vector = shap_values.values[0]

    feature_impacts = []
    for col, transformed_val, impact in zip(feature_names, X_transformed_df.iloc[0], shap_vector):
        raw_col_name = col.split("__")[-1]
        raw_val = X.iloc[0].get(raw_col_name, transformed_val)
        if isinstance(raw_val, (np.integer, np.floating)):
            raw_val = raw_val.item()
        elif pd.isna(raw_val):
            raw_val = None
        feature_impacts.append({
            "feature": raw_col_name,
            "label": FEATURE_LABELS.get(raw_col_name, raw_col_name.replace("_", " ").title()),
            "value": raw_val,
            "impact": float(impact),
        })

    feature_impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
    top_factors = feature_impacts[:5]

    return {
        "risk_score": float(np.round(prob, 4)),
        "risk_band": risk_band,
        "top_risk_factors": top_factors,
        "model_version": metadata.get("selected_model_version", "v1"),
        "model_type": metadata.get("selected_model_name", "LightGBM"),
    }


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
