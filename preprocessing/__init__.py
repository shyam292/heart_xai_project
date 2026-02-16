"""
Preprocessing Pipeline Orchestrator.

Coordinates all preprocessing steps in the correct order:
  1. Missing value imputation
  2. Outlier removal (IQR)
  3. Stratified train-test split
  4. Z-score standardization
  5. SMOTE oversampling (training set only)
  6. PCA dimensionality reduction

This module also provides a ``preprocess_single_input`` function for
real-time predictions in the Streamlit dashboard.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.imputation import impute_missing_values
from preprocessing.outlier import remove_outliers_iqr
from preprocessing.scaling import scale_features
from preprocessing.smote import apply_smote
from preprocessing.pca import apply_pca
from utils.helpers import (
    load_dataset, FEATURE_NAMES, CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES, TARGET_COLUMN,
)


def run_pipeline(test_size: float = 0.2,
                 random_state: int = 42,
                 apply_pca_transform: bool = True,
                 pca_variance: float = 0.95) -> dict:
    """
    Execute the full preprocessing pipeline end-to-end.

    Parameters
    ----------
    test_size : float
        Fraction of data reserved for testing.
    random_state : int
        Random seed for reproducibility.
    apply_pca_transform : bool
        Whether to apply PCA. When False, returns scaled data without PCA
        (useful for SHAP/LIME which need original feature names).
    pca_variance : float
        Variance retained by PCA (only used if apply_pca_transform=True).

    Returns
    -------
    dict
        Dictionary containing all pipeline outputs:
        - X_train, X_test, y_train, y_test (final processed data)
        - X_train_scaled, X_test_scaled (pre-PCA scaled data for XAI)
        - scaler, pca_model (fitted transformers)
        - df_clean (cleaned dataframe before split)
        - feature_names (list of feature names used)
    """
    print("=" * 60)
    print("  PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1 — Load raw data
    print("\n[1/6] Loading dataset...")
    df = load_dataset()
    print(f"  Loaded {len(df)} samples with {len(df.columns)} columns")

    # Step 2 — Impute missing values
    print("\n[2/6] Imputing missing values...")
    df = impute_missing_values(df, NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
    missing = df.isnull().sum().sum()
    print(f"  Remaining missing values: {missing}")

    # Step 3 — Remove outliers
    print("\n[3/6] Removing outliers (IQR method)...")
    df_clean = remove_outliers_iqr(df, NUMERICAL_FEATURES)
    print(f"  Clean dataset size: {len(df_clean)} samples")

    # Step 4 — Separate features and target
    X = df_clean[FEATURE_NAMES]
    y = df_clean[TARGET_COLUMN]

    # Step 5 — Stratified train-test split (80/20)
    print("\n[4/6] Stratified train-test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Train class dist: {y_train.value_counts().to_dict()}")
    print(f"  Test class dist:  {y_test.value_counts().to_dict()}")

    # Step 6 — Z-score standardization
    print("\n[5/6] Applying Z-score standardization...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Save pre-PCA scaled data (for SHAP/LIME interpretability)
    X_train_for_xai = X_train_scaled.copy()
    X_test_for_xai = X_test_scaled.copy()

    # Step 7 — SMOTE (training data only)
    print("\n[6/6] Applying SMOTE for class balancing...")
    X_train_balanced, y_train_balanced = apply_smote(
        X_train_scaled, y_train, random_state=random_state
    )

    # Build result dictionary
    result = {
        "X_train_scaled": X_train_for_xai,  # Pre-SMOTE scaled (for XAI)
        "X_test_scaled": X_test_for_xai,
        "scaler": scaler,
        "df_clean": df_clean,
        "feature_names": FEATURE_NAMES,
        "y_train_original": y_train,
        "y_test": y_test,
    }

    # Optional PCA
    if apply_pca_transform:
        print("\n[Bonus] Applying PCA dimensionality reduction...")
        X_train_pca, X_test_pca, pca_model = apply_pca(
            X_train_balanced, X_test_scaled, variance_threshold=pca_variance
        )
        result.update({
            "X_train": X_train_pca,
            "X_test": X_test_pca,
            "y_train": y_train_balanced,
            "pca_model": pca_model,
        })
    else:
        result.update({
            "X_train": X_train_balanced,
            "X_test": X_test_scaled,
            "y_train": y_train_balanced,
            "pca_model": None,
        })

    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE")
    print("=" * 60)

    return result


def preprocess_single_input(input_dict: dict,
                            scaler,
                            pca_model=None) -> np.ndarray:
    """
    Preprocess a single patient input for real-time prediction.

    Used by the Streamlit dashboard to transform user inputs
    before passing them to the trained model.

    Parameters
    ----------
    input_dict : dict
        Dictionary mapping feature names to values.
    scaler : StandardScaler
        Fitted scaler from the training pipeline.
    pca_model : PCA, optional
        Fitted PCA model. If None, returns scaled features directly.

    Returns
    -------
    np.ndarray
        Preprocessed feature array ready for model.predict().
    """
    input_df = pd.DataFrame([input_dict], columns=FEATURE_NAMES)
    input_scaled = scaler.transform(input_df)

    if pca_model is not None:
        input_transformed = pca_model.transform(input_scaled)
        return input_transformed
    return input_scaled
