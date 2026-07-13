"""
Unit and integration tests for the Machine Learning Risk Scoring module.
Verifies preprocess, predict, explain, tool wrapper, and fairness audit flows.
"""

import os
from pathlib import Path
import pytest
import pandas as pd
from src.risk_model.config import ModelConfig
from src.risk_model.preprocess import (
    load_raw_data,
    prepare_pipeline_data,
    split_data,
    get_preprocessor,
)
from src.risk_model.predict import run_applicant_inference
from src.risk_model.shap_explain import get_attribution_explanation
from src.tools.risk_scoring_tool import score_applicant_risk
from src.risk_model.fairness import run_fairness_analysis


def test_preprocess_data_loading_and_preparation():
    config = ModelConfig()
    
    # Test raw data load
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert config.ID_COLUMN in df.columns
    
    # Test pipeline data preparation
    X, y, application_ids = prepare_pipeline_data(df, config)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert isinstance(application_ids, pd.Series)
    assert len(X) == len(df)
    
    # Verify all configured features exist in X
    for col in config.NUMERICAL_FEATURES:
        if col in df.columns:
            assert col in X.columns
            
    for col in config.CATEGORICAL_FEATURES:
        if col in df.columns:
            assert col in X.columns
            
    # Verify engineered features exist in X
    engineered_cols = [
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "CREDIT_ANNUITY_RATIO",
        "CREDIT_GOODS_RATIO",
        "AGE_YEARS",
        "EMPLOYMENT_YEARS",
        "CHILDREN_RATIO",
        "INCOME_PER_PERSON",
        "EXT_SOURCE_MEAN",
        "EXT_SOURCE_STD",
        "EXT_SOURCE_MAX",
        "EXT_SOURCE_MIN",
        "TOTAL_MISSING_VALUES",
        "MISSING_PERCENTAGE",
        "TOTAL_DOCUMENT_FLAGS"
    ]
    for col in engineered_cols:
        assert col in X.columns


def test_preprocess_splitting_and_transformer():
    config = ModelConfig()
    df = load_raw_data()
    X, y, _ = prepare_pipeline_data(df, config)
    
    # Test train-test split
    X_train, X_test, y_train, y_test = split_data(X, y, config)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) + len(X_test) == len(X)
    
    # Test preprocessor instantiation and fitting
    preprocessor = get_preprocessor(config)
    preprocessor.fit(X_train)
    X_trans = preprocessor.transform(X_test)
    assert X_trans.shape[0] == X_test.shape[0]
    assert X_trans.shape[1] > X_test.shape[1]  # Expanded due to one-hot encoding


def test_production_inference():
    # Retrieve a sample application ID from the dataset
    df = load_raw_data()
    config = ModelConfig()
    sample_id = int(df[config.ID_COLUMN].iloc[0])
    
    # Run prediction
    result = run_applicant_inference(sample_id)
    
    # Verify response schema
    assert result["application_id"] == sample_id
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["risk_band"] in {"LOW", "MEDIUM", "HIGH"}
    assert isinstance(result["top_risk_factors"], list)
    assert len(result["top_risk_factors"]) <= 5
    assert "v1" in result["model_version"]
    assert "LightGBM" in result["model_type"]


def test_explainability_attribution():
    df = load_raw_data()
    config = ModelConfig()
    sample_id = int(df[config.ID_COLUMN].iloc[0])
    
    # Run local explanation
    explanation = get_attribution_explanation(sample_id)
    
    assert explanation["application_id"] == sample_id
    assert isinstance(explanation["top_risk_drivers"], list)
    assert len(explanation["top_risk_drivers"]) <= 5
    for driver in explanation["top_risk_drivers"]:
        assert "feature" in driver
        assert "value" in driver
        assert "impact" in driver


def test_risk_scoring_underwriting_tool():
    df = load_raw_data()
    config = ModelConfig()
    sample_id = int(df[config.ID_COLUMN].iloc[0])
    
    # Run underwriting tool wrapper
    tool_result = score_applicant_risk(sample_id)
    
    assert tool_result["application_id"] == sample_id
    assert 0.0 <= tool_result["risk_score"] <= 1.0
    assert tool_result["risk_band"] in {"LOW", "MEDIUM", "HIGH"}
    assert isinstance(tool_result["top_risk_factors"], list)
    assert len(tool_result["top_risk_factors"]) <= 5
    assert isinstance(tool_result["decision_support"], str)
    assert len(tool_result["decision_support"]) > 0
    assert "v1" in tool_result["model_version"]


def test_fairness_audit_run():
    # Run fairness audit
    df_fairness, csv_path = run_fairness_analysis()
    
    assert isinstance(df_fairness, pd.DataFrame)
    assert csv_path.exists()
    assert len(df_fairness) > 0
    
    # Verify CSV has expected columns
    expected_cols = [
        "attribute", "subgroup", "sample_size", "approval_rate",
        "high_risk_rate", "precision", "recall",
        "false_positive_rate", "false_negative_rate"
    ]
    for col in expected_cols:
        assert col in df_fairness.columns
