# streamlit_app/app.py
"""
InsightVio - Streamlit Dashboard (Live Input + Upload CSV)
Author: Nikul Ram
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add root path so Streamlit can find insightvio/ and core/
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Imports
from insightvio.explainer import InsightExplainer
from insightvio.visualizer import InsightVisualizer
from insightvio.model_trainer import train_random_forest
from core.preprocessor import preprocess_dataset
from insightvio.data_loader import load_breast_cancer_data, load_csv_data

# UI config
st.set_page_config(page_title="InsightVio", layout="wide")
st.title("InsightVio - Explainable AI Visual Toolkit")

# Mode: CSV Upload or Live Input
mode = st.radio("Select Mode", ["Upload CSV", "Live Input"])

# -------------------------
# Mode 1: CSV Upload
# -------------------------
if mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())

        label_column = st.selectbox("Select the label column", options=df.columns)

        # Infer task type if not explicitly selected
        label_dtype = df[label_column].dtype
        inferred_task = "regression" if np.issubdtype(label_dtype, np.floating) else "classification"
        task_type = st.radio("Select task type", options=["classification", "regression"],
                             index=0 if inferred_task == "classification" else 1)

        method = st.radio("Select explanation method", options=["SHAP", "LIME"])
        row_index = st.slider("Select a row to explain", 0, len(df) - 1, 0)

        if st.button("Run Explanation"):
            try:
                X, y, feature_names, _ = preprocess_dataset(df.copy(), label_column)
                model = train_random_forest(X, y, task=task_type)
                explainer = InsightExplainer(model, X, feature_names, task=task_type)
                visualizer = InsightVisualizer()

                instance = X[row_index].reshape(1, -1)
                prediction = model.predict(instance)[0]

                st.subheader(f"Explanation for Row #{row_index}")
                st.markdown(f"**Predicted {'Value' if task_type == 'regression' else 'Class'}:** `{prediction}`")

                if method == "SHAP":
                    raw_shap = explainer.shap_explainer(instance)

                    # Handle SHAP output shape safely
                    if isinstance(raw_shap.values, list):  # multiclass
                        predicted_class = int(np.argmax(model.predict(instance)))
                        shap_values = raw_shap.values[predicted_class][0]
                    elif raw_shap.values.ndim == 3:  # (1, n_features, 2)
                        predicted_class = int(np.argmax(model.predict(instance)))
                        shap_values = raw_shap.values[0, :, predicted_class]
                    else:
                        shap_values = raw_shap.values[0] if raw_shap.values.shape[0] == 1 else raw_shap.values

                    fig = visualizer.plot_shap_explanation(shap_values, feature_names, instance[0])
                    st.pyplot(fig)

                elif method == "LIME":
                    lime_exp = explainer.explain_with_lime(instance[0])
                    fig = visualizer.plot_lime_explanation(lime_exp)
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"An error occurred: {e}")

# -------------------------
# Mode 2: Live Input
# -------------------------
elif mode == "Live Input":
    st.subheader("Live Model Explanation (Manual Input)")
    dataset_name = st.selectbox("Choose dataset base:", ["breast_cancer"])

    if dataset_name == "breast_cancer":
        X_train, X_test, y_train, y_test, feature_names = load_breast_cancer_data()
        model = train_random_forest(X_train, y_train, task="classification")
        explainer = InsightExplainer(model, X_train, feature_names, task="classification")
        visualizer = InsightVisualizer()
    else:
        st.warning("Only breast_cancer supported for now.")
        st.stop()

    st.markdown("### Enter values for prediction")
    user_input = []
    for fname in feature_names:
        default = float(np.mean(X_train[:, list(feature_names).index(fname)]))
        value = st.number_input(f"{fname}", value=default)
        user_input.append(value)

    instance = np.array(user_input).reshape(1, -1)
    method = st.radio("Select explanation method", options=["SHAP", "LIME"])

    if st.button("Run Live Explanation"):
        try:
            prediction = model.predict(instance)[0]
            st.markdown(f"**Predicted Class:** `{prediction}`")

            if method == "SHAP":
                raw_shap = explainer.shap_explainer(instance)

                if isinstance(raw_shap.values, list):  # multiclass
                    predicted_class = int(np.argmax(model.predict(instance)))
                    shap_values = raw_shap.values[predicted_class][0]
                elif raw_shap.values.ndim == 3:  # binary, shape (1, n_features, 2)
                    predicted_class = int(np.argmax(model.predict(instance)))
                    shap_values = raw_shap.values[0, :, predicted_class]
                else:
                    shap_values = raw_shap.values[0] if raw_shap.values.shape[0] == 1 else raw_shap.values

                fig = visualizer.plot_shap_explanation(shap_values, feature_names, instance[0])
                st.pyplot(fig)

            elif method == "LIME":
                lime_exp = explainer.explain_with_lime(instance[0])
                fig = visualizer.plot_lime_explanation(lime_exp)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {e}")
