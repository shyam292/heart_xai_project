"""
PCA Dimensionality Reduction Module.

Applies Principal Component Analysis retaining 95% of total variance.
This reduces feature dimensionality while preserving the information
that matters most for prediction.

The PCA transformer is fit on training data only and applied to both sets.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def apply_pca(X_train: pd.DataFrame,
              X_test: pd.DataFrame,
              variance_threshold: float = 0.95) -> tuple:
    """
    Apply PCA to reduce dimensionality while retaining specified variance.

    Parameters
    ----------
    X_train : pd.DataFrame
        Scaled training feature matrix.
    X_test : pd.DataFrame
        Scaled test feature matrix.
    variance_threshold : float, optional
        Fraction of total variance to retain. Default 0.95.

    Returns
    -------
    tuple
        (X_train_pca, X_test_pca, pca_model)
        - Transformed DataFrames with PC columns
        - Fitted PCA object
    """
    pca = PCA(n_components=variance_threshold, random_state=42)

    # Fit on training data only
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    n_components = pca.n_components_
    explained_var = sum(pca.explained_variance_ratio_) * 100

    pc_columns = [f"PC{i + 1}" for i in range(n_components)]

    X_train_pca = pd.DataFrame(X_train_pca, columns=pc_columns)
    X_test_pca = pd.DataFrame(X_test_pca, columns=pc_columns)

    print(f"  [PCA] Reduced {X_train.shape[1]} features → {n_components} components")
    print(f"  [PCA] Explained variance: {explained_var:.1f}%")

    return X_train_pca, X_test_pca, pca
