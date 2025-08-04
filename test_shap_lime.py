# test_shap_lime.py
import shap
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

# Load sample data
data = load_breast_cancer()
X = data.data
y = data.target

# Train a simple model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Explain with SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Plot SHAP summary
if isinstance(shap_values, list):
    shap.summary_plot(shap_values[0], X)
else:
    shap.summary_plot(shap_values, X)


print("SHAP test successful.")

# Explain with LIME
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    X, feature_names=data.feature_names, class_names=data.target_names, discretize_continuous=True
)

explanation = explainer_lime.explain_instance(X[0], model.predict_proba, num_features=5)
explanation.as_html()
with open("lime_output.html", "w", encoding="utf-8") as f:
    f.write(explanation.as_html())


print("LIME test successful.")
