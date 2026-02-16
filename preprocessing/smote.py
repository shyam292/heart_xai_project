"""
SMOTE Oversampling Module.

Applies Synthetic Minority Over-sampling Technique (SMOTE) to handle
class imbalance in the training set. SMOTE generates synthetic samples
for the minority class by interpolating between existing minority
instances, improving model sensitivity to under-represented cases.

IMPORTANT: SMOTE is applied ONLY to training data, never to test data.
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE


def apply_smote(X_train: pd.DataFrame,
                y_train: pd.Series,
                random_state: int = 42) -> tuple:
    """
    Apply SMOTE to balance the training dataset.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target labels.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (X_resampled, y_resampled) with balanced class distribution.
    """
    # Record original distribution
    original_counts = y_train.value_counts().to_dict()
    print(f"  [SMOTE] Original class distribution: {original_counts}")

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Convert back to DataFrame/Series for consistency
    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_resampled = pd.Series(y_resampled, name=y_train.name)

    new_counts = y_resampled.value_counts().to_dict()
    print(f"  [SMOTE] Resampled class distribution: {new_counts}")
    print(f"  [SMOTE] Samples added: {len(X_resampled) - len(X_train)}")

    return X_resampled, y_resampled
