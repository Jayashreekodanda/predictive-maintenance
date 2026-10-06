# Predictive Maintenance

A machine learning-based predictive maintenance system for detecting equipment failures and evaluating models for edge deployment.


## Overview

Predictive maintenance uses machine learning to identify potential equipment failures before they occur. This project explores multiple machine learning approaches and compares their performance for predictive maintenance applications.

The project includes data preprocessing, feature engineering, dimensionality reduction, model training, explainability, and benchmarking for edge deployment.


## Models

The project evaluates three machine learning approaches:

- Random Forest
- XGBoost
- Convolutional Neural Network (CNN)

The trained models are stored in the `models/` directory.


## Project Workflow

The main workflow includes:

1. Data preprocessing
2. Exploratory data analysis
3. Feature engineering
4. Data scaling
5. Handling class imbalance using SMOTE
6. Principal Component Analysis (PCA)
7. Model training
8. Model evaluation
9. Model explainability using SHAP and LIME
10. Edge deployment benchmarking


## Project Structure

```text
predictive-maintenance/
│
├── data/
│   ├── X_test_pca.npy
│   ├── X_test_scaled.npy
│   └── y_test.npy
│
├── models/
│   ├── rf_model.pkl
│   ├── xgb_model.json
│   └── cnn_model.h5
│
├── results/
│   └── Benchmark and evaluation results
│
├── scripts/
│   ├── benchmark.py
│   ├── combine_outputs.py
│   └── run_benchmarks.sh
│
├── benchmark_results.csv
├── dockerfile
├── requirements.txt
└── README.md

```
## Benchmarking

The project includes scripts for benchmarking the trained models for edge deployment.
The benchmarking results are stored in:
benchmark_results.csv

Additional benchmark outputs are available in the results.
Technologies Used
- Python
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- NumPy
- Pandas
- SHAP
- LIME
- Docker


## Explainability

Model explainability techniques including SHAP and LIME are explored to help understand the predictions made by the machine learning models.


## Installation
Clone the repository:
git clone https://github.com/Jayashreekodanda/predictive-maintenance.git
cd predictive-maintenance

Install the required dependencies:
pip install -r requirements.txt


## Running the Benchmark
The benchmarking scripts are available in the scripts/ directory.
python scripts/benchmark.py

For the complete benchmarking workflow:
bash scripts/run_benchmarks.sh


## Docker
A Docker configuration is included in the repository for containerized execution.
Purpose
The aim of this project is to investigate machine learning approaches for predictive maintenance and evaluate their suitability for deployment in resource-constrained edge environments.


## Author
Jayashree Kodanda
GitHub: https://github.com/Jayashreekodanda