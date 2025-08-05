# model_trainer.py
"""
InsightVio - Model Training Module
Author: Nikul Ram
"""

from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X, y):
    """
    Trains a RandomForest classifier.
    Returns:
        Trained model
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model
