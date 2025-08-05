# insightvio/model_trainer.py
"""
InsightVio - Model Trainer
Author: Nikul Ram

Supports training classifiers and regressors using RandomForest.
"""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def train_random_forest(X, y, task="classification"):
    """
    Trains a Random Forest model.
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Labels
        task (str): 'classification' or 'regression'
    Returns:
        model: Trained sklearn model
    """
    if task == "regression":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X, y)
    return model
