"""
Feature Scaling Module.

Applies Z-score (standard) normalization to ensure all features have
zero mean and unit variance. This is critical for:
- Logistic Regression (gradient-based optimization)
- PCA (variance-based dimensionality reduction)
- Distance-based comparisons in LIME
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def scale_features(X_train: pd.DataFrame,
                   X_test: pd.DataFrame) -> tuple:
    """
    Apply Z-score standardization to train and test sets.

    The scaler is fit ONLY on training data to prevent data leakage,
    then applied to both training and test sets.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    X_test : pd.DataFrame
        Test feature matrix.

    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, scaler)
        - Scaled DataFrames preserving column names
        - Fitted StandardScaler object for inverse transforms
    """
    scaler = StandardScaler()

    # Fit on training data only — prevents data leakage
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    print(f"  [Scaling] Z-score standardization applied to {len(X_train.columns)} features")

    return X_train_scaled, X_test_scaled, scaler
