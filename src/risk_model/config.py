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

DATA_PATH = "data/home_credit_data/application_train.csv"
DATA_PATH_ABS = PROJECT_ROOT / DATA_PATH

# Target and ID columns requested by user
TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"


class ModelConfig:
    """Configuration class for ML model hyperparameters and pipeline parameters."""

    # Feature columns (selected for MVP from Home Credit dataset)
    #
    # Assumption A-8b (PRD, Confirmed in Sprint 0): CODE_GENDER and DAYS_BIRTH
    # must never be used as ML predictor features, only retained for fairness
    # auditing (see src/api/services/scoring.py and fairness.py). AGE_YEARS is
    # a pure linear transform of DAYS_BIRTH (engineer_features()) and carries
    # the same banned signal, so it is excluded here too - it is still computed
    # for fairness age-band bucketing, just not fed to the model.
    NUMERICAL_FEATURES: List[str] = [
        # Raw Numerical Columns
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

        # Engineered Demographic Features (age excluded per A-8b, see note above)
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
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
    ]

    # Attributes retained ONLY for fairness auditing (fairness.py) - never
    # passed to the model as predictors. Age band is derived from DAYS_BIRTH.
    FAIRNESS_ONLY_ATTRIBUTES: List[str] = [
        "CODE_GENDER",
        "DAYS_BIRTH",
    ]
    
    TARGET_COLUMN: str = TARGET_COLUMN
    ID_COLUMN: str = ID_COLUMN
    
    # Training hyperparameters
    HYPERPARAMETERS: Dict[str, Any] = {
        "n_estimators": 400,
        "max_depth": 8,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }
    
    # Validation settings
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    def __repr__(self) -> str:
        return f"<ModelConfig target={self.TARGET_COLUMN}>"

