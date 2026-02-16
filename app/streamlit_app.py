"""
Streamlit Dashboard — Explainable AI for Heart Disease Prediction.

A professional, multi-page web interface that integrates:
  📊 Dataset Overview — feature distributions & class balance
  🤖 Model Performance — metrics comparison & ROC curves
  ❤️ Prediction Interface — patient input, prediction, reasons & suggestions
  🔍 Explainability — SHAP & LIME visual explanations
  📈 Ethical Insights — bias awareness & clinical integration notes
"""

import sys
import os

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc

from utils.helpers import (
    MODELS_DIR, FEATURE_NAMES, TARGET_COLUMN,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, load_dataset,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease XAI Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a clean, professional look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global */
    .main .block-container { padding-top: 1.5rem; max-width: 1200px; }
    h1 { color: #c0392b; }
    h2 { color: #2c3e50; border-bottom: 2px solid #e74c3c; padding-bottom: 0.3rem; }
    h3 { color: #34495e; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card h3 { color: white; margin: 0; font-size: 1.8rem; }
    .metric-card p { margin: 0; opacity: 0.9; font-size: 0.9rem; }

    /* Info boxes */
    .info-box {
        background: #f8f9fa;
        border-left: 4px solid #3498db;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .reason-card {
        background: #fff;
        border: 1px solid #e0e0e0;
        border-left: 4px solid #e74c3c;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.4rem 0;
    }
    .reason-card.positive {
        border-left-color: #2ecc71;
    }
    .suggestion-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
        border-left: 4px solid #3498db;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.4rem 0;
    }

    /* Sidebar */
    [data-testid=stSidebar] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    [data-testid=stSidebar] * { color: white !important; }
    [data-testid=stSidebar] .stRadio label { font-size: 1.05rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Health suggestion knowledge base
# ---------------------------------------------------------------------------
FEATURE_EXPLANATIONS = {
    "age": {
        "high_risk": "Older age is associated with increased cardiovascular risk due to cumulative wear on blood vessels and heart muscles.",
        "suggestion": "💊 Regular cardiovascular check-ups become more important with age. Maintain an active lifestyle with age-appropriate exercise (e.g., walking 30 min/day). Monitor blood pressure and cholesterol annually.",
    },
    "sex": {
        "high_risk": "Biological sex influences heart disease risk — males tend to develop heart disease earlier, while post-menopausal women face increased risk.",
        "suggestion": "🏥 Be aware of sex-specific risk factors. Women should discuss hormone-related risks with their doctor. Men should begin screening earlier.",
    },
    "cp": {
        "high_risk": "Asymptomatic chest pain type (type 3) is paradoxically more associated with heart disease — absence of typical angina can delay diagnosis.",
        "suggestion": "⚠️ Don't ignore atypical symptoms. Even if chest pain is absent, fatigue, shortness of breath, or discomfort during exertion should be evaluated. Schedule a stress test if recommended.",
    },
    "trestbps": {
        "high_risk": "Elevated resting blood pressure (hypertension) puts continuous strain on the heart and damages arterial walls over time.",
        "suggestion": "🧂 **Reduce sodium intake** to <2,300 mg/day. Practice stress-reduction techniques (meditation, deep breathing). Exercise regularly. If on medication, maintain consistent dosing. Target BP: <120/80 mmHg.",
    },
    "chol": {
        "high_risk": "High serum cholesterol contributes to plaque buildup (atherosclerosis), narrowing arteries and restricting blood flow to the heart.",
        "suggestion": "🥗 **Adopt a heart-healthy diet**: Increase fiber (oats, beans), eat omega-3 rich foods (salmon, walnuts), reduce saturated fats (fried foods, red meat). Target total cholesterol <200 mg/dL. Consider statins if diet alone is insufficient.",
    },
    "fbs": {
        "high_risk": "Elevated fasting blood sugar (>120 mg/dL) indicates potential diabetes, which significantly increases cardiovascular risk through vascular damage.",
        "suggestion": "🍎 **Manage blood sugar**: Reduce refined carbohydrate and sugar intake. Include whole grains, vegetables, and lean proteins. Monitor HbA1c levels. Maintain a healthy weight. Consult an endocrinologist if persistently elevated.",
    },
    "restecg": {
        "high_risk": "Abnormal resting ECG results (ST-T wave abnormalities or ventricular hypertrophy) may indicate existing cardiac structural or electrical issues.",
        "suggestion": "📋 Follow up with a cardiologist for further evaluation (echocardiogram, Holter monitor). Avoid self-medicating. Report any palpitations, dizziness, or fainting episodes.",
    },
    "thalach": {
        "high_risk": "Lower maximum heart rate during exercise suggests reduced cardiac fitness and the heart's diminished ability to respond to physical demands.",
        "suggestion": "🏃 **Improve cardiovascular fitness gradually**: Start with light aerobic exercise (walking, swimming). Aim for 150 min/week of moderate activity. A cardiac rehabilitation program may help if heart disease is present.",
    },
    "exang": {
        "high_risk": "Exercise-induced angina indicates that the heart muscle isn't receiving enough blood during physical stress, a hallmark sign of coronary artery disease.",
        "suggestion": "🚨 **Seek medical evaluation immediately** if you experience chest pain during exercise. Avoid strenuous activity until cleared by a cardiologist. A stress test or angiogram may be recommended.",
    },
    "oldpeak": {
        "high_risk": "ST depression during exercise reflects myocardial ischemia — the heart muscle is starved of oxygen during physical activity.",
        "suggestion": "📊 This is a clinical indicator requiring medical follow-up. A higher oldpeak value warrants further diagnostic testing (coronary angiography). Follow your cardiologist's treatment plan.",
    },
    "slope": {
        "high_risk": "A flat or downsloping ST segment during peak exercise is a stronger indicator of coronary artery disease compared to upsloping.",
        "suggestion": "🩺 Discuss your exercise ECG results with your cardiologist. Downsloping patterns may indicate multi-vessel disease. Further imaging (CT angiography) may be needed.",
    },
    "ca": {
        "high_risk": "More major vessels colored by fluoroscopy indicates more extensive coronary artery disease — more arteries are narrowed or blocked.",
        "suggestion": "💉 This finding typically comes from catheterization. Follow your interventional cardiologist's recommendations regarding stenting, bypass surgery, or aggressive medical management.",
    },
    "thal": {
        "high_risk": "Reversible defects in thallium stress testing indicate areas of the heart that aren't getting adequate blood flow during stress but recover at rest — a sign of significant coronary artery disease.",
        "suggestion": "🔬 A reversible defect is a key finding. Follow up with your cardiologist for treatment options. Lifestyle modifications (diet, exercise, smoking cessation) and medication adherence are crucial.",
    },
}


def _get_model_display_name(model) -> str:
    """Get a human-readable name from the model object."""
    cls = type(model).__name__
    mapping = {
        "LogisticRegression": "Logistic Regression",
        "RandomForestClassifier": "Random Forest",
        "XGBClassifier": "XGBoost",
    }
    return mapping.get(cls, cls)


def _generate_reasons_and_suggestions(input_data: dict, shap_values_instance,
                                       feature_names: list, prediction: int):
    """
    Generate human-readable reasons and health suggestions based on
    SHAP feature contributions for a single prediction.

    Returns
    -------
    tuple of (list[dict], list[dict])
        reasons: list of dicts with keys 'feature', 'direction', 'explanation'
        suggestions: list of dicts with keys 'feature', 'suggestion'
    """
    # Pair features with their SHAP values
    contributions = list(zip(feature_names, shap_values_instance))
    # Sort by absolute value (most important first)
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    reasons = []
    suggestions = []

    for feature, shap_val in contributions:
        abs_val = abs(shap_val)
        if abs_val < 0.01:
            continue  # skip negligible contributions

        # Direction: positive SHAP pushes toward disease (class 1)
        if shap_val > 0:
            direction = "risk_increasing"
            direction_label = "⬆️ Increases Risk"
        else:
            direction = "risk_decreasing"
            direction_label = "⬇️ Decreases Risk"

        # Feature value context
        value = input_data.get(feature, "N/A")

        # Get explanation from knowledge base
        feat_info = FEATURE_EXPLANATIONS.get(feature, {})
        explanation = feat_info.get("high_risk", f"This feature ({feature}) influenced the prediction.")
        suggestion = feat_info.get("suggestion", "")

        reasons.append({
            "feature": feature,
            "value": value,
            "shap_value": shap_val,
            "direction": direction_label,
            "explanation": explanation,
            "abs_importance": abs_val,
        })

        if direction == "risk_increasing" and suggestion and prediction == 1:
            suggestions.append({
                "feature": feature,
                "suggestion": suggestion,
            })

    return reasons, suggestions


# ---------------------------------------------------------------------------
# Data & model loading helpers (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_raw_data():
    """Load the raw dataset for visualization."""
    return load_dataset()


@st.cache_data
def load_xai_data():
    """Load preprocessed XAI data saved during training."""
    path = os.path.join(MODELS_DIR, "xai_data.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_data
def load_eval_results():
    """Load evaluation results DataFrame."""
    path = os.path.join(MODELS_DIR, "evaluation_results.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_resource
def load_model(filename):
    """Load a trained model from disk."""
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_resource
def load_scaler():
    """Load the fitted StandardScaler."""
    path = os.path.join(MODELS_DIR, "scaler.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown("# ❤️ Heart XAI")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Dataset Overview",
        "🤖 Model Performance",
        "❤️ Prediction",
        "🔍 Explainability",
        "📈 Ethical Insights",
    ],
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>Explainable AI for Healthcare<br>"
    "Research Prototype v1.0</small>",
    unsafe_allow_html=True,
)


# =========================================================================
# PAGE 1: Dataset Overview
# =========================================================================
if page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")
    st.markdown(
        "Explore the **UCI Heart Disease dataset** — 303 patient records with "
        "13 clinical features and a binary target (heart disease vs. no disease)."
    )

    df = load_raw_data()
    if df is None:
        st.error("Dataset not found. Please place `heart.csv` in the `data/` folder.")
        st.stop()

    # --- Key statistics ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="metric-card"><h3>{len(df)}</h3><p>Total Samples</p></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card"><h3>{df.shape[1] - 1}</h3><p>Features</p></div>',
            unsafe_allow_html=True,
        )
    with col3:
        disease_pct = (df[TARGET_COLUMN].sum() / len(df) * 100)
        st.markdown(
            f'<div class="metric-card"><h3>{disease_pct:.1f}%</h3><p>Disease Positive</p></div>',
            unsafe_allow_html=True,
        )
    with col4:
        missing = df.isnull().sum().sum()
        st.markdown(
            f'<div class="metric-card"><h3>{missing}</h3><p>Missing Values</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # --- Class distribution ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Class Distribution")
        target_counts = df[TARGET_COLUMN].value_counts()
        fig = px.pie(
            values=target_counts.values,
            names=["No Disease (0)", "Heart Disease (1)"],
            color_discrete_sequence=["#2ecc71", "#e74c3c"],
            hole=0.4,
        )
        fig.update_layout(
            margin=dict(t=30, b=30, l=30, r=30),
            font=dict(size=14),
        )
        st.plotly_chart(fig, key="pie_class_dist")

    with col_right:
        st.subheader("Feature Correlation Heatmap")
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        corr = df.corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, square=True,
            linewidths=0.5, ax=ax_corr, cbar_kws={"shrink": 0.8},
        )
        ax_corr.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_corr)
        plt.close(fig_corr)

    # --- Feature distributions ---
    st.subheader("Feature Distributions")
    selected_feature = st.selectbox("Select a feature to explore", FEATURE_NAMES)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_hist = px.histogram(
            df, x=selected_feature, color=TARGET_COLUMN,
            barmode="overlay", nbins=30,
            color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
            labels={TARGET_COLUMN: "Heart Disease"},
            opacity=0.7,
        )
        fig_hist.update_layout(
            title=f"Distribution of {selected_feature}",
            margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig_hist, key="hist_feature")

    with col_b:
        fig_box = px.box(
            df, x=TARGET_COLUMN, y=selected_feature,
            color=TARGET_COLUMN,
            color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
            labels={TARGET_COLUMN: "Heart Disease"},
        )
        fig_box.update_layout(
            title=f"Box Plot of {selected_feature} by Class",
            margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig_box, key="box_feature")

    # --- Raw data preview ---
    with st.expander("📋 View Raw Data"):
        st.dataframe(df, use_container_width=True)
        st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")


# =========================================================================
# PAGE 2: Model Performance
# =========================================================================
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance Comparison")
    st.markdown(
        "Compare **Logistic Regression**, **Random Forest**, and **XGBoost** "
        "across multiple evaluation metrics."
    )

    results_df = load_eval_results()
    if results_df is None:
        st.warning(
            "⚠️ No evaluation results found. Please run `python training/train_models.py` first."
        )
        st.stop()

    # --- Metrics table ---
    st.subheader("📊 Evaluation Metrics")
    display_df = results_df[["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]].copy()
    best_idx = display_df["ROC-AUC"].idxmax()
    st.dataframe(
        display_df.style.highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
            color="#d4edda",
        ),
        use_container_width=True,
        hide_index=True,
    )

    best_model_name = display_df.loc[best_idx, "Model"]
    st.markdown(
        f'<div class="success-box">★ <strong>Best Model:</strong> {best_model_name} '
        f'(ROC-AUC: {display_df.loc[best_idx, "ROC-AUC"]:.4f})</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # --- Metrics bar chart ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Metric Comparison")
        metrics_long = display_df.melt(
            id_vars="Model",
            value_vars=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
            var_name="Metric",
            value_name="Score",
        )
        fig_bar = px.bar(
            metrics_long, x="Metric", y="Score", color="Model",
            barmode="group",
            color_discrete_sequence=["#3498db", "#2ecc71", "#e74c3c"],
        )
        fig_bar.update_layout(
            yaxis_range=[0, 1.05],
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig_bar, key="bar_metrics")

    with col2:
        st.subheader("ROC Curves")
        xai_data = load_xai_data()
        if xai_data is not None:
            fig_roc = go.Figure()
            model_files = {
                "Logistic Regression": "logistic_regression.joblib",
                "Random Forest": "random_forest.joblib",
                "XGBoost": "xgboost.joblib",
            }
            colors = ["#3498db", "#2ecc71", "#e74c3c"]

            for (name, fname), color in zip(model_files.items(), colors):
                model = load_model(fname)
                if model is not None:
                    try:
                        # Try with XAI data (non-PCA)
                        y_proba = model.predict_proba(xai_data["X_test"])[:, 1]
                        fpr, tpr, _ = roc_curve(xai_data["y_test"], y_proba)
                        roc_auc = auc(fpr, tpr)
                        fig_roc.add_trace(go.Scatter(
                            x=fpr, y=tpr, name=f"{name} (AUC={roc_auc:.3f})",
                            line=dict(color=color, width=2),
                        ))
                    except Exception:
                        pass

            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], name="Random Baseline",
                line=dict(color="gray", width=1, dash="dash"),
            ))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                margin=dict(t=30, b=30),
                legend=dict(x=0.4, y=0.1),
            )
            st.plotly_chart(fig_roc, key="roc_curves")

    # --- Confusion matrices ---
    st.subheader("Confusion Matrices")
    cm_cols = st.columns(len(results_df))
    for idx, row in results_df.iterrows():
        with cm_cols[idx]:
            cm = np.array(row["Confusion Matrix"])
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"],
                ax=ax_cm, cbar=False,
            )
            ax_cm.set_xlabel("Predicted", fontsize=10)
            ax_cm.set_ylabel("Actual", fontsize=10)
            ax_cm.set_title(row["Model"], fontsize=11, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)


# =========================================================================
# PAGE 3: Prediction Interface with Reasons & Suggestions
# =========================================================================
elif page == "❤️ Prediction":
    st.title("❤️ Heart Disease Prediction")
    st.markdown(
        "Enter patient clinical features below to get a **real-time prediction** "
        "with **reasons for the outcome** and **personalized health suggestions**."
    )

    model = load_model("best_model_xai.joblib")
    scaler = load_scaler()
    xai_data = load_xai_data()

    if model is None or scaler is None:
        st.warning("⚠️ Model not found. Please run `python training/train_models.py` first.")
        st.stop()

    model_display = _get_model_display_name(model)
    st.markdown(
        f'<div class="info-box">🤖 <strong>Active Model:</strong> {model_display}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("🩺 Patient Features")

    # Input form with two columns
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age (years)", 20, 100, 55)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox(
            "Chest Pain Type",
            [0, 1, 2, 3],
            format_func=lambda x: {
                0: "Typical Angina",
                1: "Atypical Angina",
                2: "Non-Anginal Pain",
                3: "Asymptomatic",
            }[x],
        )
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 130)
        chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 240)
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )

    with col2:
        restecg = st.selectbox(
            "Resting ECG",
            [0, 1, 2],
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T Abnormality",
                2: "Left Ventricular Hypertrophy",
            }[x],
        )
        thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox(
            "Exercise-Induced Angina",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
        )
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, step=0.1)
        slope = st.selectbox(
            "Slope of Peak Exercise ST",
            [0, 1, 2],
            format_func=lambda x: {
                0: "Upsloping", 1: "Flat", 2: "Downsloping"
            }[x],
        )
        ca = st.selectbox("Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
        thal = st.selectbox(
            "Thalassemia",
            [0, 1, 2, 3],
            format_func=lambda x: {
                0: "Normal", 1: "Fixed Defect",
                2: "Reversible Defect", 3: "Reversible Defect (severe)",
            }[x],
        )

    st.markdown("---")

    # Predict button
    if st.button("🔍 Predict", type="primary"):
        input_data = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope,
            "ca": ca, "thal": thal,
        }

        input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        prob_disease = probability[1] * 100
        prob_no_disease = probability[0] * 100

        st.markdown("---")

        # ----------------------------------------------------------
        # Result display
        # ----------------------------------------------------------
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            if prediction == 1:
                st.error("### ⚠️ Heart Disease Detected")
                st.markdown(
                    f'<div class="metric-card" style="background: '
                    f'linear-gradient(135deg, #e74c3c, #c0392b);">'
                    f'<h3>{prob_disease:.1f}%</h3>'
                    f'<p>Probability of Heart Disease</p></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.success("### ✅ No Heart Disease Detected")
                st.markdown(
                    f'<div class="metric-card" style="background: '
                    f'linear-gradient(135deg, #2ecc71, #27ae60);">'
                    f'<h3>{prob_no_disease:.1f}%</h3>'
                    f'<p>Probability of No Disease</p></div>',
                    unsafe_allow_html=True,
                )

        with res_col2:
            st.subheader("Confidence Breakdown")
            fig_conf = go.Figure(go.Bar(
                x=[prob_no_disease, prob_disease],
                y=["No Disease", "Heart Disease"],
                orientation="h",
                marker_color=["#2ecc71", "#e74c3c"],
                text=[f"{prob_no_disease:.1f}%", f"{prob_disease:.1f}%"],
                textposition="auto",
            ))
            fig_conf.update_layout(
                xaxis_title="Probability (%)",
                xaxis_range=[0, 105],
                margin=dict(t=10, b=30, l=30, r=30),
                height=200,
            )
            st.plotly_chart(fig_conf, key="conf_breakdown")

        # ----------------------------------------------------------
        # SHAP-based Reasons & Suggestions
        # ----------------------------------------------------------
        st.markdown("---")
        st.subheader("📋 Why This Prediction?")
        st.markdown(
            '<div class="info-box">'
            "The following analysis uses <strong>SHAP</strong> (SHapley Additive exPlanations) "
            "to identify which patient features most influenced this prediction and in which direction."
            "</div>",
            unsafe_allow_html=True,
        )

        # Compute SHAP for this single instance
        try:
            from explainability.shap_explainer import get_shap_explainer, compute_shap_values

            # Use training data as background for the explainer
            if xai_data is not None:
                X_bg = xai_data["X_train"]
            else:
                X_bg = input_df  # fallback

            with st.spinner("Analyzing feature contributions..."):
                explainer = get_shap_explainer(model, X_bg)
                shap_vals = compute_shap_values(explainer, pd.DataFrame(input_scaled, columns=FEATURE_NAMES))
                shap_instance = shap_vals[0] if shap_vals.ndim == 2 else shap_vals

            reasons, suggestions = _generate_reasons_and_suggestions(
                input_data, shap_instance, FEATURE_NAMES, prediction
            )

            # ----- Top Contributing Factors -----
            if reasons:
                st.markdown("#### 🔑 Key Contributing Factors")
                st.markdown(
                    "_Features are ranked by their impact on the prediction. "
                    "⬆️ means the feature pushes toward disease; ⬇️ means it pushes away._"
                )

                # Show top 6 reasons
                for i, r in enumerate(reasons[:6]):
                    css_class = "reason-card" if r["direction"].startswith("⬆️") else "reason-card positive"
                    st.markdown(
                        f'<div class="{css_class}">'
                        f'<strong>{i+1}. {r["feature"]}</strong> '
                        f'(value: {r["value"]}) — {r["direction"]}<br>'
                        f'<small>{r["explanation"]}</small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # SHAP contribution bar chart
                st.markdown("#### 📊 Feature Impact Chart")
                top_reasons = reasons[:10]
                fig_reasons = go.Figure(go.Bar(
                    x=[r["shap_value"] for r in top_reasons],
                    y=[r["feature"] for r in top_reasons],
                    orientation="h",
                    marker_color=[
                        "#e74c3c" if r["shap_value"] > 0 else "#2ecc71"
                        for r in top_reasons
                    ],
                    text=[f"{r['shap_value']:+.3f}" for r in top_reasons],
                    textposition="auto",
                ))
                fig_reasons.update_layout(
                    xaxis_title="SHAP Value (impact on prediction)",
                    yaxis=dict(autorange="reversed"),
                    margin=dict(t=10, b=40, l=10, r=10),
                    height=350,
                )
                st.plotly_chart(fig_reasons, key="reasons_chart")

            # ----- Health Suggestions -----
            if prediction == 1 and suggestions:
                st.markdown("---")
                st.subheader("💡 Health Improvement Suggestions")
                st.markdown(
                    '<div class="warning-box">'
                    "⚠️ <strong>Important:</strong> These suggestions are AI-generated "
                    "based on feature analysis and general medical knowledge. "
                    "They are <strong>NOT</strong> a substitute for professional "
                    "medical advice. Please consult your doctor."
                    "</div>",
                    unsafe_allow_html=True,
                )

                for s in suggestions:
                    st.markdown(
                        f'<div class="suggestion-card">'
                        f'<strong>Regarding: {s["feature"]}</strong><br>'
                        f'{s["suggestion"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # General lifestyle suggestions
                st.markdown("#### 🌟 General Heart-Healthy Lifestyle Tips")
                general_tips = [
                    "🚭 **Quit smoking** — Smoking is the single largest preventable risk factor for heart disease.",
                    "🏃 **Exercise regularly** — Aim for at least 150 minutes of moderate aerobic activity per week.",
                    "🥗 **Eat a balanced diet** — Rich in fruits, vegetables, whole grains, lean proteins, and omega-3 fatty acids.",
                    "⚖️ **Maintain healthy weight** — BMI between 18.5–24.9 reduces cardiovascular strain.",
                    "😴 **Sleep 7–9 hours** — Poor sleep is linked to higher blood pressure and heart disease risk.",
                    "🧘 **Manage stress** — Chronic stress contributes to high blood pressure and unhealthy coping behaviors.",
                    "🍷 **Limit alcohol** — Excessive alcohol increases blood pressure and adds empty calories.",
                ]
                for tip in general_tips:
                    st.markdown(f"- {tip}")

            elif prediction == 0:
                st.markdown("---")
                st.subheader("✅ Good News!")
                st.markdown(
                    '<div class="success-box">'
                    "Based on the provided features, the model predicts <strong>low risk</strong> "
                    "of heart disease. Continue maintaining a healthy lifestyle with regular "
                    "check-ups, balanced nutrition, and physical activity."
                    "</div>",
                    unsafe_allow_html=True,
                )

        except Exception as e:
            st.warning(
                f"⚠️ Could not compute feature explanations: {e}\n\n"
                "The prediction is still valid; only the reasons/suggestions section could not be generated."
            )

        st.markdown(
            '<div class="warning-box">'
            "⚠️ <strong>Disclaimer:</strong> This prediction is for research "
            "purposes only. Always consult a qualified healthcare professional "
            "for medical decisions.</div>",
            unsafe_allow_html=True,
        )


# =========================================================================
# PAGE 4: Explainability Dashboard
# =========================================================================
elif page == "🔍 Explainability":
    st.title("🔍 Explainability Dashboard")
    st.markdown(
        "Understand **why** the model makes its predictions using "
        "**SHAP** and **LIME** explainability techniques."
    )

    model = load_model("best_model_xai.joblib")
    xai_data = load_xai_data()
    scaler = load_scaler()

    if model is None or xai_data is None:
        st.warning("⚠️ Model / data not found. Please run `python training/train_models.py` first.")
        st.stop()

    X_train = xai_data["X_train"]
    X_test = xai_data["X_test"]
    y_test = xai_data["y_test"]
    feature_names = xai_data["feature_names"]

    model_display = _get_model_display_name(model)
    st.markdown(
        f'<div class="info-box">🤖 <strong>Explaining model:</strong> {model_display}</div>',
        unsafe_allow_html=True,
    )

    tab_shap, tab_lime = st.tabs(["🔷 SHAP Explanations", "🟢 LIME Explanations"])

    # --- SHAP Tab ---
    with tab_shap:
        st.subheader("SHAP — Global Feature Importance")
        st.markdown(
            '<div class="info-box">'
            "<strong>SHAP</strong> (SHapley Additive exPlanations) uses game theory "
            "to assign each feature an importance value. Higher |SHAP| means the "
            "feature has a stronger influence on the prediction."
            "</div>",
            unsafe_allow_html=True,
        )

        with st.spinner("Computing SHAP values (this may take a moment)..."):
            from explainability.shap_explainer import (
                get_shap_explainer, compute_shap_values,
                plot_global_importance, plot_summary, plot_force_single,
            )

            # Auto-detects model type — no hardcoded model_name needed
            explainer = get_shap_explainer(model, X_train)
            shap_values = compute_shap_values(explainer, X_test)

        # Global importance
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_imp = plot_global_importance(shap_values, feature_names)
            st.pyplot(fig_imp)
            plt.close(fig_imp)

        with col_g2:
            fig_sum = plot_summary(shap_values, X_test)
            st.pyplot(fig_sum)
            plt.close(fig_sum)

        # Local explanation
        st.markdown("---")
        st.subheader("SHAP — Individual Prediction Explanation")

        sample_idx = st.slider(
            "Select a test sample index",
            0, len(X_test) - 1, 0,
            key="shap_sample",
        )
        instance = X_test.iloc[sample_idx]
        actual = y_test.iloc[sample_idx]
        pred = model.predict(instance.values.reshape(1, -1))[0]

        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.info(f"**Actual Label:** {'Heart Disease' if actual == 1 else 'No Disease'}")
        with col_info2:
            st.info(f"**Predicted Label:** {'Heart Disease' if pred == 1 else 'No Disease'}")

        fig_water = plot_force_single(
            explainer, shap_values[sample_idx], instance, feature_names
        )
        st.pyplot(fig_water)
        plt.close(fig_water)

    # --- LIME Tab ---
    with tab_lime:
        st.subheader("LIME — Local Instance Explanation")
        st.markdown(
            '<div class="info-box">'
            "<strong>LIME</strong> (Local Interpretable Model-agnostic Explanations) "
            "creates a simpler model around each prediction to reveal "
            "which features contributed most to pushing toward one class or the other."
            "</div>",
            unsafe_allow_html=True,
        )

        sample_idx_lime = st.slider(
            "Select a test sample index",
            0, len(X_test) - 1, 0,
            key="lime_sample",
        )

        with st.spinner("Computing LIME explanation..."):
            from explainability.lime_explainer import (
                create_lime_explainer, explain_instance,
                plot_lime_explanation, get_feature_contributions,
            )

            lime_explainer = create_lime_explainer(X_train, feature_names)
            instance_lime = X_test.iloc[sample_idx_lime].values
            explanation = explain_instance(lime_explainer, model, instance_lime)

        actual_lime = y_test.iloc[sample_idx_lime]
        pred_lime = model.predict(instance_lime.reshape(1, -1))[0]

        col_li1, col_li2 = st.columns(2)
        with col_li1:
            st.info(f"**Actual Label:** {'Heart Disease' if actual_lime == 1 else 'No Disease'}")
        with col_li2:
            st.info(f"**Predicted Label:** {'Heart Disease' if pred_lime == 1 else 'No Disease'}")

        # LIME plot
        fig_lime = plot_lime_explanation(
            explanation, title=f"LIME Explanation — Sample #{sample_idx_lime}"
        )
        st.pyplot(fig_lime)
        plt.close(fig_lime)

        # Feature contribution table
        st.subheader("Feature Contribution Breakdown")
        contrib_df = get_feature_contributions(explanation)
        st.dataframe(
            contrib_df[["Feature", "Contribution", "Direction"]],
            use_container_width=True,
            hide_index=True,
        )


# =========================================================================
# PAGE 5: Ethical Insights
# =========================================================================
elif page == "📈 Ethical Insights":
    st.title("📈 Ethical Insights & Clinical Considerations")
    st.markdown(
        "Responsible AI deployment in healthcare requires awareness of "
        "**biases, limitations, and ethical considerations**."
    )

    st.markdown("---")

    # --- Bias Awareness ---
    st.subheader("⚖️ Bias Awareness")
    st.markdown("""
    <div class="warning-box">
    <strong>Dataset Bias Considerations:</strong>
    <ul>
        <li><strong>Geographic bias:</strong> The UCI Heart Disease dataset originates from
        the Cleveland Clinic and may not represent global populations.</li>
        <li><strong>Gender imbalance:</strong> The dataset contains more male patients,
        potentially reducing model accuracy for female patients.</li>
        <li><strong>Age distribution:</strong> The dataset skews toward middle-aged and
        older adults. Predictions for younger patients may be less reliable.</li>
        <li><strong>Socioeconomic factors:</strong> The dataset does not capture
        socioeconomic determinants of health.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # Show dataset demographics
    df = load_raw_data()
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            fig_sex = px.histogram(
                df, x="sex", color=TARGET_COLUMN,
                barmode="group",
                labels={"sex": "Sex (0=Female, 1=Male)"},
                color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                title="Gender Distribution by Outcome",
            )
            st.plotly_chart(fig_sex, key="eth_gender")

        with col2:
            fig_age = px.histogram(
                df, x="age", color=TARGET_COLUMN,
                barmode="overlay", nbins=20, opacity=0.7,
                color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                title="Age Distribution by Outcome",
            )
            st.plotly_chart(fig_age, key="eth_age")

    st.markdown("---")

    # --- Model Limitations ---
    st.subheader("🔒 Model Limitations")
    st.markdown("""
    <div class="info-box">
    <strong>Important limitations to consider:</strong>
    <ul>
        <li><strong>Small dataset:</strong> Only 303 samples — insufficient for
        robust generalization in clinical settings.</li>
        <li><strong>Binary classification:</strong> Heart disease severity is a spectrum,
        not a binary outcome. This model simplifies clinical reality.</li>
        <li><strong>Feature limitations:</strong> Only 13 features are used. Modern
        clinical decision-making considers imaging, genetics, and lifestyle factors.</li>
        <li><strong>No temporal data:</strong> The model does not account for disease
        progression over time.</li>
        <li><strong>No external validation:</strong> The model has not been validated
        on an independent dataset from a different institution.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Clinical Integration Notes ---
    st.subheader("🏥 Clinical Integration Guidance")
    st.markdown("""
    <div class="success-box">
    <strong>For responsible clinical integration:</strong>
    <ul>
        <li><strong>Decision support, not replacement:</strong> This system should
        augment clinician judgment, not replace it.</li>
        <li><strong>Explainability as dialogue:</strong> Use SHAP/LIME explanations
        to facilitate conversations between clinicians and patients.</li>
        <li><strong>Continuous monitoring:</strong> Model performance should be
        monitored for drift when deployed on new patient populations.</li>
        <li><strong>Regulatory compliance:</strong> Any clinical deployment must
        follow relevant regulations (e.g., FDA, EU MDR, HIPAA).</li>
        <li><strong>Informed consent:</strong> Patients should be informed when
        AI-assisted tools influence their diagnosis.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Research Ethics ---
    st.subheader("📚 Research Ethics & Transparency")
    st.markdown("""
    This project follows key principles of **responsible AI research**:

    | Principle | Implementation |
    |-----------|---------------|
    | **Transparency** | Full model architecture and training process disclosed |
    | **Explainability** | SHAP & LIME provide interpretable predictions |
    | **Reproducibility** | Fixed random seeds, saved models, documented pipeline |
    | **Fairness** | Bias analysis provided; limitations openly acknowledged |
    | **Privacy** | Uses publicly available anonymized dataset |

    > *"The right to explanation is a cornerstone of trustworthy AI in healthcare."*
    > — EU Ethics Guidelines for Trustworthy AI (2019)
    """)
