# Medical Disease Prediction Using Machine Learning

## Project Overview

This project presents a **Machine Learning-based Medical Disease Prediction System** designed to predict the likelihood of diseases from structured clinical data. The system benchmarks multiple classification algorithms and compares their predictive performance using standard evaluation metrics.

The implementation follows a modular machine learning pipeline that includes data preprocessing, hyperparameter tuning, model training, evaluation, and comparative analysis.

---

## Problem Statement

Develop an intelligent machine learning framework capable of predicting the presence of diseases using patient medical attributes while comparing multiple classification algorithms to identify the best-performing model.

---

## Objectives

* Build an end-to-end disease prediction pipeline.
* Perform automated data preprocessing and feature scaling.
* Train and optimize multiple machine learning models.
* Compare model performance using standard evaluation metrics.
* Identify the most suitable algorithm for disease prediction.

---

## Datasets

The project evaluates three structured medical datasets representing different disease domains:

| Dataset       | Description                                                              |
| ------------- | ------------------------------------------------------------------------ |
| Heart Disease | Patient demographics, cholesterol, blood pressure, ECG, heart rate, etc. |
| Diabetes      | Glucose level, BMI, insulin, blood pressure, age, etc.                   |
| Breast Cancer | Clinical laboratory measurements and diagnostic features                 |

> **Note:** The current implementation generates synthetic datasets to demonstrate the complete machine learning workflow. The pipeline can be directly integrated with real UCI or Kaggle medical datasets.

---

## Machine Learning Models

The system benchmarks the following classification algorithms:

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* XGBoost

Hyperparameter optimization is performed using **GridSearchCV** with 5-fold cross-validation.

---

## Machine Learning Pipeline

The prediction pipeline consists of the following stages:

1. Data Loading
2. Missing Value Imputation
3. Feature Standardization
4. Train-Test Split
5. Hyperparameter Optimization
6. Model Training
7. Prediction
8. Performance Evaluation

---

## Performance Metrics

Each model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Confusion Matrix

These metrics provide a comprehensive assessment of model performance for binary disease classification.

---

## Technology Stack

* Python 3
* NumPy
* Pandas
* Scikit-learn
* XGBoost

---

## Project Structure

```text
Medical-Disease-Prediction/
│
├── medical_prediction.py
├── README.md
└── requirements.txt
```

---

## Installation

Install the required dependencies:

```bash
pip install numpy pandas scikit-learn xgboost
```

---

## Running the Project

Execute the application using:

```bash
python medical_prediction.py
```

---

## Output

The application generates:

* Model-wise performance comparison
* Best hyperparameters
* Accuracy, Precision, Recall, F1-Score, and ROC-AUC
* Confusion Matrix
* Comparative performance report for each disease dataset

---

## Key Features

* Automated machine learning workflow
* Modular preprocessing pipeline
* Hyperparameter tuning with GridSearchCV
* Multi-model benchmarking
* Comprehensive performance evaluation
* Easily extensible to real-world healthcare datasets

---

## Future Enhancements

* Integration with real clinical datasets
* Disease risk probability visualization
* Web-based prediction interface using Flask or Streamlit
* Explainable AI (SHAP/LIME)
* Deep Learning-based disease prediction
* Multi-disease prediction in a single interface

---

## Author

**Lucky Rahangdale**
B.Tech in Artificial Intelligence

---

## Acknowledgements

* Scikit-learn
* XGBoost
* NumPy
* Pandas
