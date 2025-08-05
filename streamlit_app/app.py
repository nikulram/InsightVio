# streamlit_app/app.py
"""
InsightVio - Streamlit Dashboard
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

# Import modules from core project
from insightvio.explainer import InsightExplainer
from insightvio.visualizer import InsightVisualizer
from insightvio.model_trainer import train_random_forest
from core.preprocessor import preprocess_dataset

# Streamlit app layout
st.set_page_config(page_title="InsightVio", layout="wide")
st.title("InsightVio - Explainable AI Visual Toolkit")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # Choose label column
    label_column = st.selectbox("Select the label column", options=df.columns)

    # Choose explanation method
    method = st.radio("Select Explanation Method", options=["SHAP", "LIME"])

    # Select a row index to explain
    row_index = st.slider("Select a row to explain", 0, len(df) - 1, 0)

    # Run explanation
    if st.button("Run Explanation"):
        try:
            # Preprocess dataset (auto drops identifier column if needed)
            X, y, feature_names = preprocess_dataset(df.copy(), label_column)
            model = train_random_forest(X, y)
            explainer = InsightExplainer(model, X, feature_names)
            visualizer = InsightVisualizer()

            instance = X[row_index].reshape(1, -1)
            predicted_class = model.predict(instance)[0]

            st.subheader(f"Explanation for Row #{row_index}")
            st.markdown(f"**Predicted Class:** `{predicted_class}`")

            if method == "SHAP":
                shap_values, _ = explainer.explain_with_shap(instance)
                fig = visualizer.plot_shap_explanation(shap_values, feature_names, X[row_index], save_path=None)
                st.pyplot(fig)

            elif method == "LIME":
                lime_exp = explainer.explain_with_lime(X[row_index])
                fig = visualizer.plot_lime_explanation(lime_exp, save_path=None)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {e}")
