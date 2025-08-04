"""
InsightVio - Explanation Visualizer
Author: Nikul Ram
"""

import matplotlib.pyplot as plt
import shap


class InsightVisualizer:
    """
    Plots LIME and SHAP explanations for model interpretability.
    """

    def __init__(self):
        plt.style.use("ggplot")

    def plot_lime_explanation(self, explanation_list):
        """
        Plots the top features from LIME explanation as a horizontal bar chart.
        """
        features, weights = zip(*explanation_list)
        plt.figure(figsize=(8, 4))
        bars = plt.barh(features, weights, color="teal")
        plt.xlabel("Weight")
        plt.title("LIME Feature Importance")
        plt.gca().invert_yaxis()
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, f"{width:.2f}", va='center')
        plt.tight_layout()
        plt.show()

    def plot_shap_explanation(self, shap_values, feature_names, instance):
        """
        Plots a SHAP bar chart for a single prediction.
        """
        plt.figure(figsize=(8, 4))
        bars = plt.barh(feature_names, shap_values, color="orange")
        plt.xlabel("SHAP Value")
        plt.title("SHAP Feature Contributions")
        plt.gca().invert_yaxis()
        for bar, val in zip(bars, shap_values):
            plt.text(val, bar.get_y() + bar.get_height()/2, f"{val:.2f}", va='center')
        plt.tight_layout()
        plt.show()

    def plot_shap_force(self, shap_explainer, instance):
        """
        Uses SHAP’s force plot to visualize explanation in notebook (HTML-based).
        """
        shap.initjs()
        shap_values = shap_explainer(instance)
        return shap.force_plot(shap_values.base_values[0], shap_values.values[0], shap_values.data[0])
