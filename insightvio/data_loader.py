# data_loader.py
"""
InsightVio - Data Loader Module
Author: Nikul Ram

Handles both built-in and custom dataset loading.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd


def load_breast_cancer_data():
    """
    Loads and splits the breast cancer dataset from sklearn.
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, feature_names


def load_csv_data(file_path: str, label_column: str):
    """
    Loads and splits a custom CSV dataset given a path and label column.
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    df = pd.read_csv(file_path)
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset.")

    y = df[label_column]
    X = df.drop(columns=[label_column])
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, feature_names
