# InsightVio

### A Transparent Visual AI Toolkit for Interpretable Decision Support

**Author:** Nikul Ram  
**Status:** Public Open Source Project – Under Active Development  
**License:** MIT  
**Repository:** https://github.com/nikulram/InsightVio

---

## Overview

**InsightVio** is an open-source Python toolkit for transforming machine learning model predictions into human-understandable insights. Built with a mission to improve AI **transparency**, **trust**, and **decision support**, InsightVio serves domains where interpretability matters most—such as **healthcare**, **finance**, and **regulatory systems**.

This project bridges the gap between complex AI logic and real-world practitioners by combining established explainable AI tools (like SHAP and LIME) with custom-built visualizers and interpretable logic paths.

---

## Key Features

- **Visual Explanation Engine**  
  Converts AI decisions into intuitive visual logic trees and annotated flowcharts.

- **Built-in Explainable AI (XAI) Methods**  
  Integrates SHAP and LIME for feature impact analysis and interpretable outputs.

- **Healthcare Ready (Phase 1)**  
  Preconfigured to work with Breast Cancer, Heart Disease, and COVID-19 datasets.

- **Modular & Research-Friendly**  
  Notebook-ready, lightweight, and open for research extensions.

- **Production-Focused Roadmap**  
  API-ready and designed for clinical or regulatory embedding.

---

## Example Usage

from insightvio.explainers import InsightExplainer
from insightvio.visualizer import InsightVisualizer

# Fit model
model.fit(X_train, y_train)

# Create explainer and visualizer
explainer = InsightExplainer(model=model, data=X_train, mode='classification')
visualizer = InsightVisualizer()

# Choose a sample to explain
instance = X_test.iloc[0:1]

# LIME explanation
lime_result = explainer.explain_instance_with_lime(instance)
visualizer.plot_lime_explanation(lime_result)

# SHAP explanation
shap_result = explainer.explain_instance_with_shap(instance)
visualizer.plot_shap_explanation(shap_result, instance.columns, instance.values[0])

---

## Target Audience

- AI Researchers and Interpretable ML Enthusiasts  
- Healthcare Analysts and Medical AI Practitioners  
- Ethics & Governance Professionals  
- Graduate / PhD Students in AI, Bioinformatics, or Policy  
- Contributors aiming for meaningful open-source impact

---

## Installation

Install Python packages:

pip install shap lime matplotlib scikit-learn pandas numpy


Recommended: Python 3.9 or newer.


---

## Roadmap

**Phase 1** – Visual Explanation Toolkit + Medical Datasets  
**Phase 2** – Regression support + Custom XAI modules  
**Phase 3** – Real-time Web API & Deployment UI  
**Phase 4** – Academic Submission to IEEE / NeurIPS XAI Track  
**Phase 5** – Auto-generated decision policies & clinical trial support

---

## Phase 1 Use Case: Healthcare AI

Example datasets supported:
- Breast Cancer Wisconsin (diagnosis)
- Cleveland Heart Disease Dataset
- COVID-19 Symptoms & Mortality

Coming soon:
- Real clinical datasets (anonymized partnerships)
- Upload-your-own-data workflow

---


## Contributing

Want to collaborate? I am open to:

- New explainability methods  
- Real-world datasets (especially medical or finance)  
- Academic collaborations (PhD or MS thesis, research partnerships)

Fork this repo and open a PR or email Nikul Ram directly.

---

## License

This project is licensed under the MIT License.  
Free for public, personal, and commercial use with attribution.  
© 2025 Nikul Ram


### Notes
- SHAP plots will appear as interactive Matplotlib figures.
- LIME explanations are saved as HTML (`lime_output.html`) and opened manually in browser.
- To generate LIME again, just run:

  python test_shap_lime.py
