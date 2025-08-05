# core/preprocessor.py
"""
InsightVio - Data Preprocessing
Author: Nikul Ram

Handles label separation, basic cleaning, and task type detection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def is_regression_task(y):
    """Returns True if the target appears continuous (i.e., regression)."""
    return pd.api.types.is_numeric_dtype(y) and y.nunique() > 10


def preprocess_dataset(df: pd.DataFrame, label_column: str):
    """
    Splits into X, y and auto-drops identifier columns like 'name' or 'ID'.
    Detects task type automatically.
    
    Returns:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        feature_names (list): List of feature column names
        is_regression (bool): True if regression task
    """
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found.")

    # Drop identifier columns if present
    for col in df.columns:
        if col.lower() in ["id", "name", "patient_id"]:
            df = df.drop(columns=[col])
            print(f"Dropping identifier column: '{col}'")
            break

    y = df[label_column]
    X = df.drop(columns=[label_column])
    feature_names = X.columns.tolist()

    # Detect and process task type
    if not is_regression_task(y):
        if y.dtype == object or y.nunique() <= 10:
            y = LabelEncoder().fit_transform(y)
        task_type = "classification"
        is_regression = False
    else:
        y = y.astype(float)
        task_type = "regression"
        is_regression = True

    print(f"Detected task type: {task_type}")
    return X.values, y, feature_names, is_regression
