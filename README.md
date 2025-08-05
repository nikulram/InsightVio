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


Recommended: Python 3.9 or newer.(I am Using Python 3.11.5)


---

## Roadmap

## Phase 1 & 2

InsightVio currently supports classification and regression tasks on real-world healthcare datasets through both CLI and Streamlit web interface. It enables full explainability via SHAP and LIME for both uploaded CSVs and manual live input.

Currently Integrated:
- Breast Cancer Wisconsin (Diagnosis) – via `load_breast_cancer()` from `sklearn.datasets`
- Parkinson’s Disease Classification – UCI ML Repository (converted to `.csv`)
- Diabetes Progression (Regression) – via `load_diabetes()` from `sklearn.datasets`
- Credit Card Fraud Detection – Kaggle Dataset (converted to `.csv`)
- Custom CSV Upload – with automatic preprocessing and label detection
- Live Manual Input Mode – supports on-the-fly explanations using form-based entry

Upcoming (Phase 3–4):
- Cleveland Heart Disease
- COVID-19 Symptoms & Mortality Outcomes
- UCI Liver Disorders Dataset
- Model upload & selection support (external pickled models)


- Test with : streamlit run streamlit_app/app.py

>All datasets are open-source and publicly available. Final README will include citations, licenses, and DOI references for proper attribution.

---

## Contributing

Interested in shaping the future of InsightVio? Collaborators are welcome for:

- Novel explainability techniques (e.g., Anchors, Integrated Gradients)
- Dataset integration (especially healthcare, finance, law)
- Academic partnerships (MS/PhD research, co-authorship, grants)

> Fork this repository and open a PR, or reach out to me **Nikul Ram** via GitHub.

---

## License

MIT License — free for personal, academic, and commercial use with attribution.

© 2025 Nikul Ram
