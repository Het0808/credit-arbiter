"""
Configuration module for the ML Risk Scoring model.

Defines all constants, file paths, feature groups, hyperparameters, and other
configuration settings required for preprocessing, training, prediction,
and explanation of the credit risk scoring model.
"""

from pathlib import Path
from typing import Dict, List, Any


# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Paths requested by user
DATA_PATH = "data/raw/application_train.csv"
MODEL_PATH = "models/risk_model.pkl"
REPORT_PATH = "reports/ml/model_metrics.json"

# Absolute paths for robustness
DATA_PATH_ABS = PROJECT_ROOT / DATA_PATH
MODEL_PATH_ABS = PROJECT_ROOT / MODEL_PATH
REPORT_PATH_ABS = PROJECT_ROOT / REPORT_PATH

# Target and ID columns requested by user
TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"


class ModelConfig:
    """Configuration class for ML model hyperparameters and pipeline parameters."""

    # Model definition
    MODEL_TYPE: str = "xgboost"  # Default model type
    
    # Feature columns (selected from Home Credit dataset).
    #
    # FAIRNESS (assumption A-8b, US-201 AC): CODE_GENDER, DAYS_BIRTH, and the
    # AGE_YEARS proxy (AGE_YEARS = -DAYS_BIRTH/365) are DELIBERATELY EXCLUDED
    # from the model feature set - they must never drive an automated decision.
    # They are still computed by engineer_features and read by the fairness
    # audit (src/risk_model/fairness.py), which is a permitted use.
    PROTECTED_EXCLUDED_FEATURES: List[str] = ["CODE_GENDER", "DAYS_BIRTH", "AGE_YEARS"]

    # Auxiliary-table aggregate features (US-201 hardening; see aux_features.py).
    # Single source of truth: NUMERICAL_FEATURES is composed from this list below,
    # and the aux module / inference path reference it directly, so they can't drift.
    AUX_FEATURES: List[str] = [
        # Bureau aggregates
        "BUREAU_LOAN_COUNT",
        "BUREAU_ACTIVE_COUNT",
        "BUREAU_AMT_CREDIT_SUM_MEAN",
        "BUREAU_AMT_CREDIT_SUM_DEBT_SUM",
        "BUREAU_CREDIT_DAY_OVERDUE_MEAN",
        "BUREAU_CREDIT_DAY_OVERDUE_MAX",
        "BUREAU_DAYS_CREDIT_MEAN",
        # Previous-application aggregates
        "PREV_APP_COUNT",
        "PREV_APP_APPROVED_RATIO",
        "PREV_APP_REFUSED_COUNT",
        "PREV_AMT_APPLICATION_MEAN",
        "PREV_AMT_CREDIT_MEAN",
        "PREV_DAYS_DECISION_MAX",
        # Payment-behaviour aggregates (installments / POS / credit card)
        "INST_COUNT",
        "INST_DPD_MEAN",
        "INST_DPD_MAX",
        "INST_LATE_COUNT",
        "INST_PAYMENT_RATIO_MEAN",
        "POS_COUNT",
        "POS_SK_DPD_MEAN",
        "POS_SK_DPD_MAX",
        "CC_COUNT",
        "CC_AMT_BALANCE_MEAN",
        "CC_UTILIZATION_MEAN",
        "CC_SK_DPD_MAX",
    ]

    # Application-table numeric features (DAYS_BIRTH & AGE_YEARS excluded - protected).
    _BASE_NUMERICAL_FEATURES: List[str] = [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "DAYS_EMPLOYED",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "CNT_FAM_MEMBERS",
        "AMT_GOODS_PRICE",
        "CNT_CHILDREN",
        # Engineered Financial Ratios
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "CREDIT_ANNUITY_RATIO",
        "CREDIT_GOODS_RATIO",
        # Engineered Non-Protected Features (AGE_YEARS excluded - age proxy)
        "EMPLOYMENT_YEARS",
        "CHILDREN_RATIO",
        "INCOME_PER_PERSON",
        # Engineered External Credit Features
        "EXT_SOURCE_MEAN",
        "EXT_SOURCE_STD",
        "EXT_SOURCE_MAX",
        "EXT_SOURCE_MIN",
        # Engineered Missing Data Features
        "TOTAL_MISSING_VALUES",
        "MISSING_PERCENTAGE",
        # Engineered Document Flags
        "TOTAL_DOCUMENT_FLAGS",
    ]

    # Full numeric feature set = application features then aux aggregates (order
    # matters: it is the order the production model was trained on).
    NUMERICAL_FEATURES: List[str] = _BASE_NUMERICAL_FEATURES + AUX_FEATURES

    CATEGORICAL_FEATURES: List[str] = [
        # CODE_GENDER excluded - protected attribute (A-8b)
        "NAME_CONTRACT_TYPE",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
    ]
    
    TARGET_COLUMN: str = TARGET_COLUMN
    ID_COLUMN: str = ID_COLUMN
    
    # Training hyperparameters
    HYPERPARAMETERS: Dict[str, Any] = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }
    
    # Validation settings
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    def __repr__(self) -> str:
        return f"<ModelConfig model_type={self.MODEL_TYPE}>"

