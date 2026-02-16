"""
Outlier Removal Module.

Uses the Interquartile Range (IQR) method to detect and remove outliers
from numerical features. Data points lying beyond 1.5 × IQR from Q1/Q3
are considered outliers and removed.
"""

import pandas as pd
import numpy as np


def remove_outliers_iqr(df: pd.DataFrame,
                        numerical_cols: list,
                        factor: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from numerical columns using the IQR method.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    numerical_cols : list
        Columns to check for outliers.
    factor : float, optional
        IQR multiplier for fence calculation. Default is 1.5.

    Returns
    -------
    pd.DataFrame
        Dataframe with outlier rows removed.
    """
    df = df.copy()
    mask = pd.Series([True] * len(df), index=df.index)

    for col in numerical_cols:
        if col not in df.columns:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        mask = mask & col_mask

    removed_count = (~mask).sum()
    if removed_count > 0:
        print(f"  [Outlier Removal] Removed {removed_count} outlier rows "
              f"({removed_count / len(df) * 100:.1f}% of data)")

    return df[mask].reset_index(drop=True)
