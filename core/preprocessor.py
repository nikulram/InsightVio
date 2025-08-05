# preprocessor.py
"""
InsightVio - Data Preprocessing Module
Author: Nikul Ram

Handles general preprocessing for custom datasets.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_dataset(df: pd.DataFrame, label_column: str):
    """
    Cleans and prepares a dataset for modeling and explanation.
    - Drops known identifier columns (like 'name')
    - Drops rows with missing labels
    - Encodes categorical features
    - Returns X, y, and feature names
    """
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found.")

    # Drop rows with missing label
    df = df.dropna(subset=[label_column])

    # ✅ Drop the 'name' column if it exists (this is causing the crash)
    if 'name' in df.columns:
        print("Dropping identifier column: 'name'")
        df = df.drop(columns=['name'])

    # Separate features and target
    y = df[label_column]
    X = df.drop(columns=[label_column])

    # Handle missing values in features
    X = X.fillna(X.mean(numeric_only=True))

    # Encode any remaining non-numeric columns
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    feature_names = X.columns.tolist()
    return X.values, y.values, feature_names
