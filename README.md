# Machine Learning vs Langmuir: A Multioutput XGBoost Regressor Better Captures Soil Phosphorus Adsorption Dynamics

## Overview

This repository contains the code and analysis supporting the manuscript.  
The data are available in the Zenodo repository: [10.5281/zenodo.15828513](https://doi.org/10.5281/zenodo.15854383).

> **Machine Learning vs Langmuir: A Multioutput XGBoost Regressor Better Captures Soil Phosphorus Adsorption Dynamics**

The study investigates the predictive power of a data-driven machine learning model (XGBoost) compared to the traditional Langmuir adsorption isotherm for estimating phosphorus (P) adsorption across a range of equilibrium concentrations.

---

## Abstract

Traditional models like the Langmuir isotherm have been widely used to describe phosphorus adsorption in soils, but they often fail to capture the full variability present in diverse soil datasets. In this work, we employ a multioutput XGBoost regressor trained on soil physico-chemical properties to predict P adsorption at multiple equilibrium concentrations. Our results demonstrate that the machine learning approach significantly outperforms the Langmuir model in predictive accuracy, offering a scalable and data-efficient tool for soil fertility diagnostics and nutrient management.

---

## Repository Contents

- `notebooks/`: Jupyter notebooks for data preprocessing, modeling, and evaluation.
- `data/`: Datasets are available on Zenodo: [10.5281/zenodo.15828513](https://doi.org/10.5281/zenodo.15854383).
- `figures/`: Plots comparing observed vs predicted values and feature importances.
- `src/`: Core Python scripts for model training and validation.
- `multioutput_xgb_model.pkl`: Trained multi-output XGBoost model (serialized with joblib).
- `README.md`: This file.

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt

---

Citation

If you use this repository in your work, please cite:

Iatrou, M.; Papadopoulos, A. Machine Learning vs. Langmuir: A Multioutput XGBoost Regressor Better Captures Soil Phosphorus Adsorption Dynamics.  
Crops 2025, 5, 55. https://doi.org/10.3390/crops5040055


---


