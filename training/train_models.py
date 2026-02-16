"""
Model Training Module.

Trains multiple classifiers on the preprocessed heart disease data:
  - Logistic Regression
  - Random Forest
  - XGBoost

Each model is trained with reproducible hyperparameters and saved to
the ``models/`` directory using joblib. The module also supports
training without PCA for explainability-compatible models.
"""

import sys
import os
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from preprocessing import run_pipeline
from training.evaluate import evaluate_all_models
from utils.helpers import MODELS_DIR, ensure_dir


def get_models() -> dict:
    """
    Return a dictionary of model name → untrained model instance.

    All models use ``random_state=42`` for full reproducibility.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        ),
    }


def train_all_models(pipeline_result: dict) -> dict:
    """
    Train all models and return fitted model objects.

    Parameters
    ----------
    pipeline_result : dict
        Output of ``run_pipeline()``.

    Returns
    -------
    dict
        Mapping of model name → fitted model.
    """
    X_train = pipeline_result["X_train"]
    y_train = pipeline_result["y_train"]

    models = get_models()
    trained_models = {}

    print("\n" + "=" * 60)
    print("  MODEL TRAINING")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"  ✓ {name} trained successfully")

    return trained_models


def save_models(trained_models: dict, best_model_name: str) -> None:
    """
    Save all trained models to disk using joblib.

    Parameters
    ----------
    trained_models : dict
        Mapping of model name → fitted model.
    best_model_name : str
        Name of the best-performing model (saved separately as best_model.joblib).
    """
    ensure_dir(MODELS_DIR)

    for name, model in trained_models.items():
        filename = name.lower().replace(" ", "_") + ".joblib"
        filepath = os.path.join(MODELS_DIR, filename)
        joblib.dump(model, filepath)
        print(f"  Saved: {filepath}")

    # Save the best model with a standard name
    best_model = trained_models[best_model_name]
    best_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_model, best_path)
    print(f"  ★ Best model ({best_model_name}) saved to: {best_path}")


def main():
    """
    Full training pipeline: preprocess → train → evaluate → save.

    This function runs two pipelines:
      1. With PCA — for model training and evaluation
      2. Without PCA — for SHAP/LIME explainability (original features)
    """
    # --- Pipeline WITH PCA (for training) ---
    print("\n▶ Running preprocessing pipeline (with PCA)...")
    pipeline_result = run_pipeline(apply_pca_transform=True)

    # --- Train models ---
    trained_models = train_all_models(pipeline_result)

    # --- Evaluate models ---
    results_df, best_name = evaluate_all_models(
        trained_models,
        pipeline_result["X_test"],
        pipeline_result["y_test"],
    )

    # --- Save models ---
    print("\n" + "=" * 60)
    print("  SAVING MODELS")
    print("=" * 60)
    save_models(trained_models, best_name)

    # --- Pipeline WITHOUT PCA (for explainability) ---
    print("\n▶ Running preprocessing pipeline (without PCA, for XAI)...")
    xai_pipeline = run_pipeline(apply_pca_transform=False)

    # Train the best model type on non-PCA data for XAI
    print(f"\n  Training {best_name} on non-PCA data for explainability...")
    xai_models = get_models()
    xai_model = xai_models[best_name]
    xai_model.fit(xai_pipeline["X_train"], xai_pipeline["y_train"])

    xai_model_path = os.path.join(MODELS_DIR, "best_model_xai.joblib")
    joblib.dump(xai_model, xai_model_path)
    print(f"  ★ XAI model saved to: {xai_model_path}")

    # Save scaler and pipeline artifacts
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    joblib.dump(xai_pipeline["scaler"], scaler_path)
    print(f"  Saved scaler to: {scaler_path}")

    if pipeline_result["pca_model"] is not None:
        pca_path = os.path.join(MODELS_DIR, "pca_model.joblib")
        joblib.dump(pipeline_result["pca_model"], pca_path)
        print(f"  Saved PCA model to: {pca_path}")

    # Save evaluation results
    results_path = os.path.join(MODELS_DIR, "evaluation_results.joblib")
    joblib.dump(results_df, results_path)
    print(f"  Saved evaluation results to: {results_path}")

    # Save the XAI pipeline data for the dashboard
    xai_data = {
        "X_train": xai_pipeline["X_train"],
        "X_test": xai_pipeline["X_test"],
        "y_train": xai_pipeline["y_train"],
        "y_test": xai_pipeline["y_test"],
        "feature_names": xai_pipeline["feature_names"],
        "df_clean": xai_pipeline["df_clean"],
    }
    xai_data_path = os.path.join(MODELS_DIR, "xai_data.joblib")
    joblib.dump(xai_data, xai_data_path)
    print(f"  Saved XAI data to: {xai_data_path}")

    print("\n" + "=" * 60)
    print("  ALL DONE! 🎉")
    print(f"  Best model: {best_name}")
    print(f"  Models saved to: {MODELS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
