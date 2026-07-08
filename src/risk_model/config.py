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
    
    # Feature columns (selected for MVP from Home Credit dataset)
    NUMERICAL_FEATURES: List[str] = [
        # Raw Numerical Columns
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "DAYS_BIRTH",
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
        
        # Engineered Demographic Features
        "AGE_YEARS",
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
    
    CATEGORICAL_FEATURES: List[str] = [
        "NAME_CONTRACT_TYPE",
        "CODE_GENDER",
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

