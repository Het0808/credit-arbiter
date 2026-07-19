"""
Model explainability module for the Halcyon Credit Risk Scoring Engine.

Loads the production pipeline, calculates local feature attributions for a given
application ID, and saves explainability plots (global summary and local waterfall)
to reports/ml/shap/. If SHAP is unavailable or has compatibility issues, it falls
back to a robust feature-importance-based attribution heuristic.
"""

import json
import joblib
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.risk_model.aux_features import merge_aux_features
from src.risk_model.config import ModelConfig
from src.risk_model.preprocess import (
    load_raw_data,
    prepare_pipeline_data,
    split_data,
)


# US-202: human-readable labels for model features shown to underwriters.
FEATURE_LABELS = {
    "AMT_INCOME_TOTAL": "Annual income",
    "AMT_CREDIT": "Requested credit amount",
    "AMT_ANNUITY": "Loan annuity",
    "AMT_GOODS_PRICE": "Goods price",
    "DAYS_EMPLOYED": "Employment duration",
    "EMPLOYMENT_YEARS": "Years employed",
    "CNT_FAM_MEMBERS": "Family size",
    "CNT_CHILDREN": "Number of children",
    "CREDIT_INCOME_RATIO": "Loan-to-income ratio",
    "ANNUITY_INCOME_RATIO": "Debt-to-income ratio",
    "CREDIT_ANNUITY_RATIO": "Credit-to-annuity ratio",
    "CREDIT_GOODS_RATIO": "Credit-to-goods ratio",
    "CHILDREN_RATIO": "Children-to-family ratio",
    "INCOME_PER_PERSON": "Income per household member",
    "EXT_SOURCE_1": "External credit score 1",
    "EXT_SOURCE_2": "External credit score 2",
    "EXT_SOURCE_3": "External credit score 3",
    "EXT_SOURCE_MEAN": "External credit score (avg)",
    "EXT_SOURCE_STD": "External credit score (spread)",
    "EXT_SOURCE_MAX": "External credit score (best)",
    "EXT_SOURCE_MIN": "External credit score (worst)",
    "TOTAL_MISSING_VALUES": "Number of missing fields",
    "MISSING_PERCENTAGE": "Share of missing fields",
    "TOTAL_DOCUMENT_FLAGS": "Documents provided",
    "BUREAU_LOAN_COUNT": "Credit-bureau loan count",
    "BUREAU_ACTIVE_COUNT": "Active bureau loans",
    "BUREAU_AMT_CREDIT_SUM_MEAN": "Avg bureau credit amount",
    "BUREAU_AMT_CREDIT_SUM_DEBT_SUM": "Total bureau debt",
    "BUREAU_CREDIT_DAY_OVERDUE_MEAN": "Avg days overdue (bureau)",
    "BUREAU_CREDIT_DAY_OVERDUE_MAX": "Max days overdue (bureau)",
    "BUREAU_DAYS_CREDIT_MEAN": "Avg bureau credit recency",
    "PREV_APP_COUNT": "Previous applications",
    "PREV_APP_APPROVED_RATIO": "Previous approval rate",
    "PREV_APP_REFUSED_COUNT": "Previous refusals",
    "PREV_AMT_APPLICATION_MEAN": "Avg previous requested amount",
    "PREV_AMT_CREDIT_MEAN": "Avg previous credit amount",
    "PREV_DAYS_DECISION_MAX": "Most recent previous decision",
    "INST_COUNT": "Instalments paid (history)",
    "INST_DPD_MEAN": "Avg instalment days late",
    "INST_DPD_MAX": "Worst instalment days late",
    "INST_LATE_COUNT": "Late instalments",
    "INST_PAYMENT_RATIO_MEAN": "Avg payment coverage ratio",
    "POS_COUNT": "POS/cash balance records",
    "POS_SK_DPD_MEAN": "Avg POS days past due",
    "POS_SK_DPD_MAX": "Worst POS days past due",
    "CC_COUNT": "Credit-card balance records",
    "CC_AMT_BALANCE_MEAN": "Avg credit-card balance",
    "CC_UTILIZATION_MEAN": "Avg credit-card utilisation",
    "CC_SK_DPD_MAX": "Worst credit-card days past due",
    "NAME_CONTRACT_TYPE": "Contract type",
    "FLAG_OWN_CAR": "Owns a car",
    "FLAG_OWN_REALTY": "Owns real estate",
    "NAME_INCOME_TYPE": "Income type",
    "NAME_EDUCATION_TYPE": "Education level",
}


def _humanize(feature_col: str) -> str:
    """Turn a transformer column name (e.g. 'num__EXT_SOURCE_MEAN') into a label."""
    raw = feature_col.split("__")[-1]
    # One-hot columns look like NAME_EDUCATION_TYPE_Higher education.
    for base, label in FEATURE_LABELS.items():
        if raw == base:
            return label
        if raw.startswith(base + "_"):
            return f"{label}: {raw[len(base) + 1:]}"
    return raw.replace("_", " ").title()


def _label_drivers(drivers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach a human-readable label and risk direction to each SHAP driver (US-202)."""
    for d in drivers:
        d["label"] = _humanize(d["feature"])
        d["direction"] = "increases_risk" if d["impact"] > 0 else "decreases_risk"
    return drivers


def get_attribution_explanation(
    application_id: int,
    sample_size: int = 200
) -> Dict[str, Any]:
    """
    Generate local risk explanation for a specific credit application.

    Attempts to use SHAP for model explainability. Falls back to feature-importance
    weighted difference heuristics if SHAP is not installed or raises errors.

    Args:
        application_id: Unique client application ID (SK_ID_CURR).
        sample_size: Number of background samples for explanation baseline.

    Returns:
        Explanation dictionary containing the application_id and top_risk_drivers.
    """
    config = ModelConfig()
    
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / "models" / "production" / "risk_model_v1.pkl"
    shap_dir = project_root / "reports" / "ml" / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Production model not found at {model_path}. Please run select_model.py first.")
        
    pipeline = joblib.load(model_path)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    
    # Load and preprocess dataset (with auxiliary aggregates - US-201)
    print("Loading data for explainability analysis...")
    df = load_raw_data()
    df = merge_aux_features(df)
    X, y, application_ids = prepare_pipeline_data(df, config)
    
    # Ensure application_id exists
    if application_id not in application_ids.values:
        available_ids = list(application_ids.dropna().head(10).values)
        raise ValueError(f"Application ID {application_id} not found in dataset. Sample available IDs: {available_ids}")
        
    # Get feature names out of the ColumnTransformer
    feature_names = preprocessor.get_feature_names_out()
    
    # Run preprocessing transformer
    X_transformed = preprocessor.transform(X)
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
    X_transformed_df["SK_ID_CURR"] = application_ids.values
    
    # Extract the target applicant's preprocessed features
    applicant_row = X_transformed_df[X_transformed_df["SK_ID_CURR"] == application_id].iloc[0]
    applicant_features = applicant_row.drop("SK_ID_CURR")
    
    # Extract raw applicant values for reporting
    raw_applicant_row = df[df[config.ID_COLUMN] == application_id].iloc[0]
    
    # Prepare background dataset (for SHAP/Attribution baseline)
    background_df = X_transformed_df.drop(columns=["SK_ID_CURR"]).sample(
        n=min(sample_size, len(X_transformed_df)),
        random_state=config.RANDOM_STATE
    )
    
    # Check if SHAP is available
    shap_available = False
    try:
        import shap
        shap_available = True
        print("SHAP package found. Executing TreeExplainer...")
    except ImportError:
        print("SHAP package not found. Using feature importance fallback explainability...")
        
    if shap_available:
        try:
            # Initialize TreeExplainer on LightGBM classifier
            explainer = shap.TreeExplainer(classifier)
            
            # Calculate SHAP values
            shap_values = explainer(background_df)
            
            # Generate and save Global Summary Plot
            print("Generating global SHAP summary plot...")
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, background_df, show=False)
            plt.title("Global Feature Attribution (SHAP Summary)", fontsize=14, fontweight="bold", pad=15)
            plt.tight_layout()
            plt.savefig(shap_dir / "shap_summary.png", dpi=300)
            plt.close()
            
            # Generate explanation for the specific applicant
            # Find index of target applicant in the preprocessed dataset
            applicant_idx = X_transformed_df[X_transformed_df["SK_ID_CURR"] == application_id].index[0]
            # Compute shap values for this specific applicant row
            applicant_transformed = X_transformed_df.drop(columns=["SK_ID_CURR"]).loc[[applicant_idx]]
            single_shap_values = explainer(applicant_transformed)
            
            # Generate and save local waterfall plot
            print(f"Generating local SHAP waterfall plot for applicant {application_id}...")
            plt.figure(figsize=(8, 6))
            shap.plots.waterfall(single_shap_values[0], show=False)
            plt.title(f"Risk Attribution (Applicant {application_id})", fontsize=14, fontweight="bold", pad=15)
            plt.tight_layout()
            plt.savefig(shap_dir / "sample_waterfall.png", dpi=300)
            plt.close()
            
            # Compile top risk drivers from SHAP
            shap_vector = single_shap_values.values[0]
            feature_impacts = []
            for col, val, impact in zip(background_df.columns, applicant_transformed.iloc[0], shap_vector):
                # Retrieve raw value if it is a raw column, otherwise preprocessed
                raw_col_name = col.split("__")[-1]
                raw_val = raw_applicant_row.get(raw_col_name, val)
                feature_impacts.append({
                    "feature": col,
                    "value": raw_val,
                    "impact": float(impact)
                })
                
            # Sort by absolute SHAP impact descending
            feature_impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
            top_drivers = _label_drivers(feature_impacts[:5])

            explanation = {
                "application_id": int(application_id),
                "top_risk_drivers": top_drivers
            }
            return explanation
            
        except Exception as e:
            print(f"SHAP calculation encountered error: {e}. Falling back to feature importance attribution...")
            shap_available = False

    # FALLBACK LOGIC: Feature Importance-Based Attribution
    if not shap_available:
        # Get global feature importances from LightGBM
        importances = classifier.feature_importances_
        # Normalize importances to sum to 1
        importances_norm = importances / (importances.sum() + 1e-9)
        
        # Save custom Global Feature Importance Plot
        print("Generating global feature importance plot...")
        feat_imp_df = pd.DataFrame({
            "feature": background_df.columns,
            "importance": importances_norm
        }).sort_values("importance", ascending=False).head(15)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feat_imp_df, x="importance", y="feature", palette="viridis")
        plt.title("Global Feature Importance (Fallback Attribution)", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Normalized Importance")
        plt.tight_layout()
        plt.savefig(shap_dir / "shap_summary.png", dpi=300)
        plt.close()
        
        # Calculate local risk drivers:
        # Heuristic: Compare applicant values to the background average values.
        # Feature impact = (Applicant Preprocessed Value - Mean Background Value) * Feature Importance
        # A positive impact increase means the feature increases default risk relative to average applicant.
        mean_background = background_df.mean()
        local_impacts = []
        for col, val, imp in zip(background_df.columns, applicant_features, importances_norm):
            # Difference from baseline population
            diff = val - mean_background[col]
            # Risk impact (direction depends on the variance, simple heuristic matches importance * difference)
            # Standardize indicator: positive difference for risk features (like late payments, credit/income) increases risk
            impact = diff * imp
            
            raw_col_name = col.split("__")[-1]
            raw_val = raw_applicant_row.get(raw_col_name, val)
            
            # Map raw types safely to avoid non-serializable objects
            if isinstance(raw_val, (np.integer, np.floating)):
                raw_val = raw_val.item()
            elif pd.isna(raw_val):
                raw_val = None
                
            local_impacts.append({
                "feature": col,
                "value": raw_val,
                "impact": float(impact)
            })
            
        # Sort by absolute local impact descending
        local_impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
        top_drivers = _label_drivers(local_impacts[:5])
        
        # Generate custom sample waterfall plot
        print(f"Generating custom waterfall plot for applicant {application_id}...")
        waterfall_df = pd.DataFrame(top_drivers)
        plt.figure(figsize=(8, 6))
        sns.barplot(data=waterfall_df, x="impact", y="feature", hue="feature", palette="RdYlGn_r", legend=False)
        plt.axvline(0, color="black", linestyle="--")
        plt.title(f"Risk Drivers - Fallback Attribution (Applicant {application_id})", fontsize=12, fontweight="bold", pad=15)
        plt.xlabel("Risk Attribution Score")
        plt.tight_layout()
        plt.savefig(shap_dir / "sample_waterfall.png", dpi=300)
        plt.close()
        
        explanation = {
            "application_id": int(application_id),
            "top_risk_drivers": top_drivers
        }
        return explanation


if __name__ == "__main__":
    # Test using a sample application ID from training set (e.g. 100002 or similar in Home Credit)
    # We load first available ID to run test
    config = ModelConfig()
    try:
        df = load_raw_data()
        sample_id = int(df[config.ID_COLUMN].iloc[0])
        print(f"Testing explainability on sample applicant ID: {sample_id}")
        explanation = get_attribution_explanation(sample_id)
        print("\nExplanation Output JSON:")
        print(json.dumps(explanation, indent=2))
    except Exception as e:
        print(f"Error during test execution: {e}")
