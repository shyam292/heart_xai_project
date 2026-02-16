"""
LIME Explainability Module.

Provides local, instance-level explanations using LIME (Local Interpretable
Model-agnostic Explanations). LIME creates a simpler, interpretable model
around each prediction to explain the behavior of the complex ML model.

Each explanation shows which features contributed most to pushing the
prediction toward one class or the other.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer


def create_lime_explainer(X_train: pd.DataFrame,
                          feature_names: list,
                          class_names: list = None) -> LimeTabularExplainer:
    """
    Create a LIME tabular explainer.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data (used as background for perturbation).
    feature_names : list
        Names of the features.
    class_names : list, optional
        Names of the target classes.

    Returns
    -------
    LimeTabularExplainer
        Configured LIME explainer.
    """
    if class_names is None:
        class_names = ["No Disease", "Heart Disease"]

    explainer = LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )
    return explainer


def explain_instance(explainer: LimeTabularExplainer,
                     model,
                     instance: np.ndarray,
                     num_features: int = 10) -> object:
    """
    Generate a LIME explanation for a single instance.

    Parameters
    ----------
    explainer : LimeTabularExplainer
        LIME explainer.
    model : estimator
        Trained classifier with ``predict_proba`` method.
    instance : np.ndarray
        1D array of feature values for the instance.
    num_features : int
        Number of top features to include in the explanation.

    Returns
    -------
    lime.explanation.Explanation
        LIME explanation object with feature weights and visualization.
    """
    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict_proba,
        num_features=num_features,
        num_samples=1000,
    )
    return explanation


def plot_lime_explanation(explanation, title: str = "LIME Explanation") -> plt.Figure:
    """
    Generate a matplotlib figure from a LIME explanation.

    Parameters
    ----------
    explanation : lime.explanation.Explanation
        LIME explanation for a single instance.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
        Bar chart showing feature contributions.
    """
    # Extract feature contributions for the predicted class
    exp_list = explanation.as_list()
    features = [x[0] for x in exp_list]
    weights = [x[1] for x in exp_list]

    # Color: green for positive contribution, red for negative
    colors = ["#e74c3c" if w > 0 else "#2ecc71" for w in weights]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(features)), weights, color=colors, edgecolor="gray", alpha=0.85)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel("Feature Contribution", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="→ Heart Disease"),
        Patch(facecolor="#2ecc71", label="→ No Disease"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def get_feature_contributions(explanation) -> pd.DataFrame:
    """
    Extract feature contributions as a DataFrame.

    Parameters
    ----------
    explanation : lime.explanation.Explanation
        LIME explanation.

    Returns
    -------
    pd.DataFrame
        DataFrame with Feature, Contribution, and Direction columns.
    """
    exp_list = explanation.as_list()
    df = pd.DataFrame(exp_list, columns=["Feature", "Contribution"])
    df["Direction"] = df["Contribution"].apply(
        lambda x: "Heart Disease ↑" if x > 0 else "No Disease ↑"
    )
    df["Abs Contribution"] = df["Contribution"].abs()
    df = df.sort_values("Abs Contribution", ascending=False)
    return df
