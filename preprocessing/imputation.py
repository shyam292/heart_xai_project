"""
Missing Value Imputation Module.

Handles missing values in the UCI Heart Disease dataset:
- Numerical features → median imputation
- Categorical features → mode imputation

This approach is robust to outliers (median) and preserves the most
frequent category distribution (mode).
"""

import pandas as pd
import numpy as np


def impute_missing_values(df: pd.DataFrame,
                          numerical_cols: list,
                          categorical_cols: list) -> pd.DataFrame:
    """
    Impute missing values in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (may contain NaN or placeholder values like '?').
    numerical_cols : list
        Column names of numerical features.
    categorical_cols : list
        Column names of categorical features.

    Returns
    -------
    pd.DataFrame
        Dataframe with missing values imputed.
    """
    df = df.copy()

    # Replace common placeholder values with NaN
    df.replace(["?", " ", ""], np.nan, inplace=True)

    # Ensure correct dtypes for numerical columns
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Numerical: median imputation ---
    for col in numerical_cols:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    # --- Categorical: mode imputation ---
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)

    return df
