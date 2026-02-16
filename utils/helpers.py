"""
Utility helpers for the Heart Disease XAI project.

Provides path constants, data loading utilities, and common functions
used across preprocessing, training, and explainability modules.
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Path constants — all paths are relative to the project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATASET_PATH = os.path.join(DATA_DIR, "heart.csv")

# Feature metadata
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

TARGET_COLUMN = "target"


def load_dataset(path: str = None) -> pd.DataFrame:
    """
    Load the UCI Heart Disease dataset from disk.

    Parameters
    ----------
    path : str, optional
        Custom path to the CSV file. Defaults to ``DATASET_PATH``.

    Returns
    -------
    pd.DataFrame
        Raw dataset with all 14 columns (13 features + target).
    """
    if path is None:
        path = DATASET_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Please place 'heart.csv' in the data/ directory."
        )
    df = pd.read_csv(path)
    return df


def ensure_dir(directory: str) -> None:
    """Create a directory if it does not already exist."""
    os.makedirs(directory, exist_ok=True)
