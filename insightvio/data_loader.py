"""
InsightVio - Data Loader Module
Author: Nikul Ram

Handles both built-in and custom dataset loading.
"""

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd

# imports custom preprocessing logic
from core.preprocessor import preprocess_dataset


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


def load_diabetes_data():
    """
    Loads and splits the diabetes regression dataset from sklearn.
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    data = load_diabetes()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, feature_names


def load_csv_data(file_path: str, label_column: str):
    """
    Loads and preprocesses a custom CSV dataset using InsightVio's preprocessor.
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    df = pd.read_csv(file_path)

    # Updated: capture 4 outputs
    X, y, feature_names, _ = preprocess_dataset(df, label_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, feature_names
