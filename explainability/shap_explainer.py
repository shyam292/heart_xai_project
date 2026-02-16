"""
SHAP Explainability Module.

Provides global and local SHAP-based explanations for heart disease
predictions. Auto-detects model type and uses the appropriate explainer:
- TreeExplainer for Random Forest, XGBoost (fast & exact)
- LinearExplainer for Logistic Regression (fast)
- KernelExplainer as fallback for any other model

SHAP (SHapley Additive exPlanations) assigns each feature an importance
value for a particular prediction, grounded in cooperative game theory.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def _detect_model_type(model) -> str:
    """
    Auto-detect the model type for choosing the right SHAP explainer.

    Returns
    -------
    str
        One of 'tree', 'linear', or 'kernel'.
    """
    model_class = type(model).__name__.lower()

    # Tree-based models
    if any(name in model_class for name in [
        "randomforest", "gradientboosting", "xgb", "lgbm",
        "decisiontree", "extratrees",
    ]):
        return "tree"

    # Linear models
    if any(name in model_class for name in [
        "logisticregression", "linearregression", "ridge",
        "lasso", "sgdclassifier", "sgdregressor",
    ]):
        return "linear"

    # Fallback
    return "kernel"


def get_shap_explainer(model, X_background: pd.DataFrame, model_name: str = ""):
    """
    Create the appropriate SHAP explainer for the given model type.

    Auto-detects the model type from the model object itself.

    Parameters
    ----------
    model : estimator
        Trained model.
    X_background : pd.DataFrame
        Background dataset for computing SHAP values (use training data).
    model_name : str
        Optional model name string (used as a hint, but auto-detection
        takes priority).

    Returns
    -------
    shap.Explainer
        Configured SHAP explainer.
    """
    model_type = _detect_model_type(model)

    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
    elif model_type == "linear":
        # LinearExplainer is fast and exact for linear models
        background = shap.sample(X_background, min(100, len(X_background)))
        explainer = shap.LinearExplainer(model, background)
    else:
        # KernelExplainer works for any model but is slower
        background = shap.sample(X_background, min(100, len(X_background)))
        explainer = shap.KernelExplainer(model.predict_proba, background)

    return explainer


def compute_shap_values(explainer, X: pd.DataFrame):
    """
    Compute SHAP values for the given data.

    Parameters
    ----------
    explainer : shap.Explainer
        SHAP explainer object.
    X : pd.DataFrame
        Feature matrix to explain.

    Returns
    -------
    np.ndarray
        SHAP values array of shape (n_samples, n_features).
        For binary classification, returns values for the positive class.
    """
    shap_values = explainer.shap_values(X)

    # Handle list of arrays (binary classification: [class_0, class_1])
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class

    shap_values = np.array(shap_values)

    # Handle 3D arrays: (n_samples, n_features, n_classes) → take class 1
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    # Ensure 2D: (n_samples, n_features)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    return shap_values


def plot_global_importance(shap_values: np.ndarray,
                           feature_names: list) -> plt.Figure:
    """
    Generate a global SHAP feature importance bar chart.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values matrix (n_samples x n_features).
    feature_names : list
        Names of the features.

    Returns
    -------
    matplotlib.figure.Figure
        Bar chart figure showing mean |SHAP| per feature.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    # Ensure 1D for DataFrame creation
    mean_abs_shap = np.array(mean_abs_shap).flatten()
    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Mean |SHAP|": mean_abs_shap,
    }).sort_values("Mean |SHAP|", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        feature_importance["Feature"],
        feature_importance["Mean |SHAP|"],
        color="#e74c3c",
        edgecolor="#c0392b",
        alpha=0.85,
    )
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("Global Feature Importance (SHAP)", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_summary(shap_values: np.ndarray,
                 X: pd.DataFrame) -> plt.Figure:
    """
    Generate a SHAP summary (beeswarm) plot.

    Shows the distribution of SHAP values for each feature, with color
    indicating feature value (red = high, blue = low).

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values matrix.
    X : pd.DataFrame
        Original feature data (for color mapping).

    Returns
    -------
    matplotlib.figure.Figure
        Summary plot figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False, plot_size=None)
    plt.title("SHAP Summary Plot", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return plt.gcf()


def plot_force_single(explainer,
                      shap_values_instance: np.ndarray,
                      instance: pd.Series,
                      feature_names: list) -> plt.Figure:
    """
    Generate a SHAP waterfall plot for a single instance.

    Shows how each feature pushes the prediction from the base value.

    Parameters
    ----------
    explainer : shap.Explainer
        SHAP explainer (needed for expected_value).
    shap_values_instance : np.ndarray
        SHAP values for the single instance (1D array).
    instance : pd.Series
        Feature values for the instance.
    feature_names : list
        Feature names.

    Returns
    -------
    matplotlib.figure.Figure
        Waterfall plot figure.
    """
    # Get expected value (base value)
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1]  # positive class

    # Create an Explanation object for the waterfall plot
    explanation = shap.Explanation(
        values=shap_values_instance,
        base_values=expected_value,
        data=instance.values if hasattr(instance, 'values') else instance,
        feature_names=feature_names,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False)
    plt.title("SHAP Waterfall — Individual Prediction", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return plt.gcf()
