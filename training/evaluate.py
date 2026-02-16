"""
Model Evaluation Module.

Computes a comprehensive set of classification metrics for each trained
model and identifies the best performer based on ROC-AUC score.

Metrics computed:
  - Accuracy
  - Precision (weighted)
  - Recall (weighted)
  - F1 Score (weighted)
  - ROC-AUC
  - Confusion Matrix
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(model, X_test, y_test) -> dict:
    """
    Compute all evaluation metrics for a single model.

    Parameters
    ----------
    model : estimator
        Trained sklearn-compatible classifier.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        True test labels.

    Returns
    -------
    dict
        Dictionary of metric name → value.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, average="weighted"), 4),
        "Recall": round(recall_score(y_test, y_pred, average="weighted"), 4),
        "F1 Score": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def evaluate_all_models(trained_models: dict,
                        X_test,
                        y_test) -> tuple:
    """
    Evaluate all trained models and return a comparison DataFrame.

    Parameters
    ----------
    trained_models : dict
        Mapping of model name → fitted model.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        True test labels.

    Returns
    -------
    tuple
        (results_df, best_model_name)
        - DataFrame with metrics for each model
        - Name of the best model by ROC-AUC
    """
    print("\n" + "=" * 60)
    print("  MODEL EVALUATION")
    print("=" * 60)

    records = []

    for name, model in trained_models.items():
        print(f"\n  Evaluating {name}...")
        metrics = compute_metrics(model, X_test, y_test)

        # Print individual report
        y_pred = model.predict(X_test)
        print(f"    Accuracy:  {metrics['Accuracy']}")
        print(f"    Precision: {metrics['Precision']}")
        print(f"    Recall:    {metrics['Recall']}")
        print(f"    F1 Score:  {metrics['F1 Score']}")
        print(f"    ROC-AUC:   {metrics['ROC-AUC']}")

        records.append({
            "Model": name,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1 Score": metrics["F1 Score"],
            "ROC-AUC": metrics["ROC-AUC"],
            "Confusion Matrix": metrics["Confusion Matrix"],
        })

    results_df = pd.DataFrame(records)

    # Identify best model by ROC-AUC
    best_idx = results_df["ROC-AUC"].idxmax()
    best_name = results_df.loc[best_idx, "Model"]

    print("\n" + "-" * 60)
    print(f"  ★ Best Model: {best_name} (ROC-AUC: {results_df.loc[best_idx, 'ROC-AUC']})")
    print("-" * 60)

    return results_df, best_name
