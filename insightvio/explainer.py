# explainer.py
"""
InsightVio - Core Explainability Engine
Author: Nikul Ram
"""

import shap
import lime.lime_tabular
import numpy as np

from sklearn.tree import _tree
from typing import List


class InsightExplainer:
    """
    Provides local and global interpretability for classification and regression models.
    """

    def __init__(self, model, X_train: np.ndarray, feature_names: List[str], task: str = "classification"):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.task = task

        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            mode=task,
            class_names=["Class 0", "Class 1"] if task == "classification" else None
        )

        self.shap_explainer = shap.Explainer(
            model.predict_proba if task == "classification" else model.predict,
            X_train
        )

    def explain_with_lime(self, instance: np.ndarray, num_features: int = 5):
        """
        Explain a prediction using LIME.
        """
        prediction_fn = self.model.predict_proba if self.task == "classification" else self.model.predict
        explanation = self.lime_explainer.explain_instance(instance, prediction_fn, num_features=num_features)
        return explanation.as_list()

    def explain_with_shap(self, instance: np.ndarray):
        """
        Explain a prediction using SHAP.
        Handles regression, binary classification, and multiclass.
        """
        shap_values = self.shap_explainer(instance)

        if isinstance(shap_values.values, np.ndarray):
            values = shap_values.values[0]
            base_value = shap_values.base_values[0]
        elif isinstance(shap_values.values, list):
            class_index = int(np.argmax(self.model.predict(instance)))
            values = shap_values.values[class_index][0]
            base_value = shap_values.base_values[class_index][0]
        else:
            raise ValueError("Unexpected SHAP values structure")

        return values, base_value

    def extract_tree_rules(self) -> List[str]:
        """
        Extract all decision rules if the model is a decision tree.
        """
        if not hasattr(self.model, 'tree_'):
            raise ValueError("Model is not a decision tree.")

        rules = []

        def recurse(node, depth, path):
            if self.model.tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = self.feature_names[self.model.tree_.feature[node]]
                threshold = self.model.tree_.threshold[node]
                recurse(self.model.tree_.children_left[node], depth + 1, path + [f"{name} <= {threshold:.2f}"])
                recurse(self.model.tree_.children_right[node], depth + 1, path + [f"{name} > {threshold:.2f}"])
            else:
                value = self.model.tree_.value[node]
                prediction = np.argmax(value[0])
                rules.append(" AND ".join(path) + f" => Predict class {prediction}")

        recurse(0, 1, [])
        return rules