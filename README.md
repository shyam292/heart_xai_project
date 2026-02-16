# Explainable AI for Transparent Decision Making in Healthcare
## Heart Disease Prediction with SHAP & LIME

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)

---

## 📌 Project Overview

This project implements a **modular Explainable AI (XAI) framework** for heart disease prediction using the UCI Heart Disease dataset. It goes beyond standard ML pipelines by integrating **SHAP** and **LIME** explainability techniques, enabling clinicians and researchers to understand *why* a model makes a particular prediction — not just *what* it predicts.

### Research Motivation

Healthcare AI systems must be **transparent and trustworthy**. Black-box models, despite high accuracy, are unsuitable for clinical adoption because:

- Clinicians need to **verify** AI reasoning against medical knowledge
- Patients have a right to **understand** decisions affecting their health
- Regulatory frameworks (e.g., EU AI Act) require **explainability** for high-risk AI

This project demonstrates how XAI techniques can bridge the gap between predictive performance and clinical interpretability.

---

## 🏗️ Architecture

```
heart_xai_project/
│
├── data/                       # Dataset storage
│   └── heart.csv
│
├── models/                     # Saved trained models (.joblib)
│
├── preprocessing/              # Modular preprocessing pipeline
│   ├── __init__.py             # Pipeline orchestrator
│   ├── imputation.py           # Missing value imputation
│   ├── outlier.py              # IQR outlier removal
│   ├── scaling.py              # Z-score standardization
│   ├── smote.py                # SMOTE oversampling
│   └── pca.py                  # PCA dimensionality reduction
│
├── training/                   # Model training & evaluation
│   ├── train_models.py         # Train LR, RF, XGBoost
│   └── evaluate.py             # Metrics computation & comparison
│
├── explainability/             # XAI modules
│   ├── shap_explainer.py       # SHAP global & local explanations
│   └── lime_explainer.py       # LIME local explanations
│
├── app/                        # Streamlit dashboard
│   └── streamlit_app.py        # Multi-page web interface
│
├── utils/                      # Utility functions
│   └── helpers.py              # Data loading, path constants
│
├── requirements.txt
└── README.md
```

---

## 🔧 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Steps

```bash
# 1. Clone or navigate to the project
cd heart_xai_project

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1: Train the Models

```bash
python training/train_models.py
```

This will:
- Load and preprocess the UCI Heart Disease dataset
- Train Logistic Regression, Random Forest, and XGBoost models
- Evaluate all models and select the best performer
- Save trained models to `models/`

### Step 2: Launch the Dashboard

```bash
streamlit run app/streamlit_app.py
```

Navigate to `http://localhost:8501` to explore:
- 📊 **Dataset Overview** — Feature distributions & class balance
- 🤖 **Model Performance** — Metrics comparison & ROC curves
- ❤️ **Prediction Interface** — Input patient data & get predictions
- 🔍 **Explainability Dashboard** — SHAP & LIME visual explanations
- 📈 **Ethical Insights** — Bias awareness & clinical considerations

---

## 📊 Dataset

**UCI Heart Disease Dataset** (Cleveland subset)
- **Samples**: 303
- **Features**: 13 clinical attributes
- **Target**: Binary (0 = No Disease, 1 = Disease)

| Feature | Description |
|---------|-------------|
| age | Age in years |
| sex | Sex (1 = male, 0 = female) |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results (0–2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy |
| thal | Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect) |

---

## 🧠 Explainability Techniques

### SHAP (SHapley Additive exPlanations)
- **Global**: Feature importance ranking across the entire dataset
- **Local**: Per-instance force plots showing how each feature pushes the prediction

### LIME (Local Interpretable Model-agnostic Explanations)
- **Local**: Per-instance explanations with feature contribution weights
- Generates interpretable linear approximations around individual predictions

---

## 📜 License

This project is released under the MIT License for academic and research purposes.

---

## 📚 References

1. Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine.
2. Lundberg, S.M. and Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
3. Ribeiro, M.T., Singh, S. and Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD*.
