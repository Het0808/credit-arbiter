"""
Preprocessing module for the ML Risk Scoring model.

Handles loading raw credit data, cleaning, missing value imputation,
categorical encoding, scaling, feature engineering, and train-test splits.
"""

from pathlib import Path
from typing import Tuple, Any, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.risk_model.config import ModelConfig, DATA_PATH_ABS, TARGET_COLUMN, ID_COLUMN


def load_raw_data(data_path: str = None) -> pd.DataFrame:
    """
    Load raw credit data from the specified path.

    If data_path is None, defaults to DATA_PATH_ABS. If not found, falls back
    to home_credit_data directory inside the repository.

    Args:
        data_path: Path to the raw CSV/parquet file.

    Returns:
        DataFrame containing the raw dataset.
    """
    if data_path is None:
        data_path = DATA_PATH_ABS
    else:
        data_path = Path(data_path)

    # Check existence
    if not data_path.exists():
        # Look for fallback in standard project structure
        project_root = Path(__file__).resolve().parents[2]
        fallback_path = project_root / "data" / "home_credit_data" / "home-credit-default-risk" / "application_train.csv"
        if fallback_path.exists():
            data_path = fallback_path
        else:
            raise FileNotFoundError(
                f"Raw dataset could not be found at {data_path} or fallback {fallback_path}."
            )

    return pd.read_csv(data_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data cleaning (deduplication, basic error corrections).

    Args:
        df: Input raw DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()
    
    # Drop complete duplicates
    df = df.drop_duplicates()
    
    # In Home Credit dataset, DAYS_EMPLOYED has 365243 representing NaN/not employed.
    # Replace it with NaN so imputer handles it correctly.
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
        
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct new features and transformations relevant to credit risk scoring.

    Includes financial ratios, demographic transformations, statistical aggregates
    for external credit sources, missing data statistics, and document flag counts.
    Handles division-by-zero safely by replacing infinite/NaN values with 0.0.

    Args:
        df: Input cleaned DataFrame.

    Returns:
        DataFrame with engineered features.
    """
    df = df.copy()

    # Define a helper for safe division
    def safe_div(numerator: pd.Series, denominator: pd.Series, fill_value: float = 0.0) -> pd.Series:
        # Avoid division-by-zero by replacing 0 with NaN, performing division, and then filling
        return numerator.div(denominator.replace(0, np.nan)).fillna(fill_value).replace([np.inf, -np.inf], fill_value)

    # 1. Financial Ratios
    df["CREDIT_INCOME_RATIO"] = safe_div(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])
    df["ANNUITY_INCOME_RATIO"] = safe_div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])
    df["CREDIT_ANNUITY_RATIO"] = safe_div(df["AMT_CREDIT"], df["AMT_ANNUITY"])
    df["CREDIT_GOODS_RATIO"] = safe_div(df["AMT_CREDIT"], df["AMT_GOODS_PRICE"])

    # 2. Demographic Features
    df["AGE_YEARS"] = df["DAYS_BIRTH"].div(-365.25)
    df["EMPLOYMENT_YEARS"] = df["DAYS_EMPLOYED"].div(-365.25)
    df["CHILDREN_RATIO"] = safe_div(df["CNT_CHILDREN"], df["CNT_FAM_MEMBERS"])
    df["INCOME_PER_PERSON"] = safe_div(df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"])

    # 3. External Credit Features
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    existing_ext = [col for col in ext_cols if col in df.columns]
    if existing_ext:
        df["EXT_SOURCE_MEAN"] = df[existing_ext].mean(axis=1)
        df["EXT_SOURCE_STD"] = df[existing_ext].std(axis=1).fillna(0.0)
        df["EXT_SOURCE_MAX"] = df[existing_ext].max(axis=1)
        df["EXT_SOURCE_MIN"] = df[existing_ext].min(axis=1)
    else:
        df["EXT_SOURCE_MEAN"] = np.nan
        df["EXT_SOURCE_STD"] = 0.0
        df["EXT_SOURCE_MAX"] = np.nan
        df["EXT_SOURCE_MIN"] = np.nan

    # 4. Missing Data Features
    # Note: We compute missing values across the original columns of df
    df["TOTAL_MISSING_VALUES"] = df.isnull().sum(axis=1).astype(float)
    df["MISSING_PERCENTAGE"] = df["TOTAL_MISSING_VALUES"].div(float(len(df.columns)))

    # 5. Document Flags
    doc_cols = [col for col in df.columns if col.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        df["TOTAL_DOCUMENT_FLAGS"] = df[doc_cols].sum(axis=1).astype(float)
    else:
        df["TOTAL_DOCUMENT_FLAGS"] = 0.0

    return df


def prepare_pipeline_data(
    df: pd.DataFrame, config: ModelConfig
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Select relevant features, engineer ratios, and split into features, target, and IDs.

    Filters the final X to only include features defined in config.py (numerical
    and categorical).

    Args:
        df: Raw input DataFrame.
        config: Configuration instance with feature names and target name.

    Returns:
        Tuple of (X, y, application_ids):
            - X: Feature matrix containing configured and engineered features.
            - y: Series containing target variable (or None if not present).
            - application_ids: Series containing application/client unique IDs.
    """
    # 1. Clean raw data
    df_cleaned = clean_data(df)

    # 2. Engineer features
    df_engineered = engineer_features(df_cleaned)

    # 2b. Merge auxiliary-table aggregates (US-201). Idempotent and a no-op if
    # already merged or if the ID column is absent, so every caller that builds
    # X gets the exact feature set the production model expects.
    from src.risk_model.aux_features import merge_aux_features

    df_engineered = merge_aux_features(df_engineered)

    # Retrieve all configured feature columns from config.py
    selected_numeric = list(config.NUMERICAL_FEATURES)
    selected_categorical = list(config.CATEGORICAL_FEATURES)
    all_features = selected_numeric + selected_categorical

    # 3. Filter DataFrame to include only selected features, target and ID
    available_features = [col for col in all_features if col in df_engineered.columns]
    X = df_engineered[available_features].copy()

    # Extract ID
    if config.ID_COLUMN in df_engineered.columns:
        application_ids = df_engineered[config.ID_COLUMN]
    else:
        application_ids = pd.Series(dtype="int64")

    # Extract Target
    if config.TARGET_COLUMN in df_engineered.columns:
        y = df_engineered[config.TARGET_COLUMN]
    else:
        y = pd.Series(dtype="float64")

    return X, y, application_ids


def get_preprocessor(config: ModelConfig) -> ColumnTransformer:
    """
    Create a sklearn ColumnTransformer for data preprocessing.

    Applies:
        - Median imputation + StandardScaler for numeric columns
        - Most frequent imputation + OneHotEncoder(handle_unknown="ignore") for categorical columns

    Args:
        config: ModelConfig instance containing feature lists.

    Returns:
        ColumnTransformer configured with preprocessing pipelines.
    """
    numeric_cols = list(config.NUMERICAL_FEATURES)
    categorical_cols = list(config.CATEGORICAL_FEATURES)

    # Numeric Pipeline
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical Pipeline
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    # Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor


def split_data(
    X: pd.DataFrame, y: pd.Series, config: ModelConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets.

    Args:
        X: Feature matrix.
        y: Target vector.
        config: ModelConfig containing split ratios and random seeds.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    # Only stratify if target variable has both classes and is binary
    stratify = y if len(y.value_counts()) > 1 else None
    
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=stratify,
    )
