# visualizer.py
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

    def plot_lime_explanation(self, explanation_list, save_path=None):
        """
        Plots the top features from LIME explanation as a horizontal bar chart.
        Handles both classification and regression explanations.
        """
        if not explanation_list or not isinstance(explanation_list[0], tuple):
            raise ValueError("LIME explanation must be a list of (feature, weight) tuples.")

        features, weights = zip(*explanation_list)
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(features, weights, color="teal")
        ax.set_xlabel("Weight")
        ax.set_title("LIME Feature Importance")
        ax.invert_yaxis()

        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2, f"{width:.2f}", va='center')

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path)
            plt.close(fig)
            return None
        else:
            return fig

    def plot_shap_explanation(self, shap_values, feature_names, instance, save_path=None):
        """
        Plots a SHAP bar chart for a single prediction.
        """
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(feature_names, shap_values, color="orange")
        ax.set_xlabel("SHAP Value")
        ax.set_title("SHAP Feature Contributions")
        ax.invert_yaxis()

        for bar, val in zip(bars, shap_values):
            ax.text(val, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va='center')

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path)
            plt.close(fig)
            return None
        else:
            return fig

    def plot_shap_force(self, shap_explainer, instance):
        """
        Uses SHAP’s force plot to visualize explanation in notebook (HTML-based).
        """
        shap.initjs()
        shap_values = shap_explainer(instance)
        return shap.force_plot(
            shap_values.base_values[0],
            shap_values.values[0],
            shap_values.data[0]
        )
