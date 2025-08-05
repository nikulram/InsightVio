# main.py
"""
InsightVio - Command-Line Interface
Author: Nikul Ram

Supports SHAP/LIME explanation on built-in or custom CSV datasets.
"""

import argparse
import numpy as np
import os

from insightvio.data_loader import load_breast_cancer_data, load_csv_data
from insightvio.model_trainer import train_random_forest
from insightvio.explainer import InsightExplainer
from insightvio.visualizer import InsightVisualizer


def run_explanation(method: str, save_path: str, csv_path: str, label_column: str):
    # Load dataset
    if csv_path and label_column:
        print(f"Loading custom dataset from {csv_path}")
        X_train, X_test, y_train, y_test, feature_names = load_csv_data(csv_path, label_column)
    else:
        print("Loading built-in breast cancer dataset")
        X_train, X_test, y_train, y_test, feature_names = load_breast_cancer_data()

    print(f"Dataset has {X_train.shape[1]} features")

    # Train model
    model = train_random_forest(X_train, y_train)
    print("RandomForest model trained successfully")

    # Create explainer + visualizer
    explainer = InsightExplainer(model, X_train, feature_names)
    visualizer = InsightVisualizer()

    # Select first test instance
    instance = X_test[0].reshape(1, -1)

    # Run explanation
    if method == "shap":
        shap_vals, _ = explainer.explain_with_shap(instance)
        visualizer.plot_shap_explanation(shap_vals, feature_names, X_test[0], save_path)
    elif method == "lime":
        lime_exp = explainer.explain_with_lime(X_test[0])
        visualizer.plot_lime_explanation(lime_exp, save_path)
    else:
        raise ValueError("Invalid method. Choose 'shap' or 'lime'.")

    if save_path:
        print(f"Plot saved to: {save_path}")
    else:
        print("Plot shown interactively")


def parse_args():
    parser = argparse.ArgumentParser(description="Run InsightVio explanation")
    parser.add_argument("--method", type=str, choices=["shap", "lime"], default="shap", help="Explanation method to use")
    parser.add_argument("--save", type=str, help="Path to save output plot (PNG)")
    parser.add_argument("--csv", type=str, help="Path to custom CSV file")
    parser.add_argument("--label", type=str, help="Label column in custom CSV file")
    return parser.parse_args()


def main():
    args = parse_args()
    run_explanation(
        method=args.method,
        save_path=args.save,
        csv_path=args.csv,
        label_column=args.label
    )


if __name__ == "__main__":
    main()
