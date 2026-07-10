import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# DYNAMIC ENGINE PACKAGE DEPLOYMENT & ALIGNMENT

try:
    from xgboost import XGBClassifier
except ImportError:
    user_site_packages = r"c:\users\lucky\appdata\roaming\python\python313\site-packages"
    if user_site_packages not in sys.path:
        sys.path.append(user_site_packages)
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("[!] XGBoost not mapped. Forcing baseline extraction setup...")
        os.system(f'"{sys.executable}" -m pip install xgboost')
        from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

class MedicalInferenceEngine:
    """
    Production-grade classification framework optimized to accept structured
    medical profiles and benchmark multiple machine learning algorithms.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_models = {}
        self.metrics_report = {}

    def _get_algorithm_matrix(self):
        """Maps specific algorithms to hyperparameter search grids."""
        return {
            'Logistic_Regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=2000),
                'params': {'classifier__C': [0.01, 0.1, 1.0, 10.0]}
            },
            'SVM': {
                'model': SVC(random_state=self.random_state, probability=True),
                'params': {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['linear', 'rbf']}
            },
            'Random_Forest': {
                'model': RandomForestClassifier(random_state=self.random_state),
                'params': {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [None, 10]}
            },
            'XGBoost': {
                'model': XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                'params': {'classifier__n_estimators': [50, 100], 'classifier__learning_rate': [0.1, 0.2]}
            }
        }

    def evaluate_dataset(self, X, y, disease_target_name):
        """Runs train-test splits, data profiling, scaling, and metrics collection."""
        print(f"\n{'='*70}\nInitializing Target Vector Analysis: {disease_target_name}\n{'='*70}")
        
        # Stratified Splitting to preserve medical diagnostic balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=self.random_state, stratify=y
        )

        algorithm_matrix = self._get_algorithm_matrix()
        compiled_results = []

        for algo_name, config in algorithm_matrix.items():
            print(f"[+] Processing Optimization Tuning Loop: {algo_name}...")
            
            # Preprocessing Pipeline 
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),  
                ('scaler', StandardScaler()),                  
                ('classifier', config['model'])
            ])
            
            grid_search = GridSearchCV(
                estimator=pipeline, 
                param_grid=config['params'], 
                cv=5, 
                scoring='accuracy', 
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            optimal_estimator = grid_search.best_estimator_
            self.best_models[f"{disease_target_name}_{algo_name}"] = optimal_estimator
            
            # Inferential Prediction Evaluation
            predictions = optimal_estimator.predict(X_test)
            probabilities = optimal_estimator.predict_proba(X_test)[:, 1]
            
            metrics = {
                'Disease Dataset': disease_target_name,
                'Algorithm Class': algo_name,
                'Accuracy': accuracy_score(y_test, predictions),
                'Precision': precision_score(y_test, predictions, zero_division=0),
                'Recall': recall_score(y_test, predictions, zero_division=0),
                'F1-Score': f1_score(y_test, predictions, zero_division=0),
                'ROC_AUC': roc_auc_score(y_test, probabilities)
            }
            compiled_results.append(metrics)
            
        self.metrics_report[disease_target_name] = pd.DataFrame(compiled_results)
        return self.metrics_report[disease_target_name]


# CLINICAL DATABASES 

if __name__ == "__main__":
    np.random.seed(42)
    
    # Dataset 1: UCI Heart Disease Profile (Symptoms, age, blood test results, resting bp)
    X_heart = pd.DataFrame(np.random.randn(300, 13), columns=[
        'age', 'sex', 'chest_pain_symptom', 'resting_bp', 'cholesterol', 'blood_sugar_test', 
        'rest_ecg', 'max_heart_rate', 'exercise_angina', 'oldpeak', 'st_slope', 'ca_vessels', 'thal'
    ])
    y_heart = np.random.choice([0, 1], size=300, p=[0.46, 0.54])
    
    # Dataset 2: UCI Diabetes Profile (Glucose, insulin levels, age, BMI metrics)
    X_diabetes = pd.DataFrame(np.random.randn(400, 8), columns=[
        'pregnancies', 'glucose_test', 'blood_pressure', 'skin_thickness', 'insulin_test', 'bmi', 'pedigree', 'age'
    ])
    y_diabetes = np.random.choice([0, 1], size=400, p=[0.65, 0.35])
    
    # Dataset 3: UCI Breast Cancer Profile (Lab test measurements, continuous features)
    X_cancer = pd.DataFrame(np.random.randn(250, 30), columns=[f'lab_result_{i}' for i in range(30)])
    y_cancer = np.random.choice([0, 1], size=250, p=[0.37, 0.63])

    # Run Analysis Workload
    engine = MedicalInferenceEngine(random_state=42)
    engine.evaluate_dataset(X_heart, y_heart, "UCI_Heart_Disease")
    engine.evaluate_dataset(X_diabetes, y_diabetes, "UCI_Diabetes")
    engine.evaluate_dataset(X_cancer, y_cancer, "UCI_Breast_Cancer")
    
    # Print Consolidated Operational Performance Matrix 
    print("\n\n" + "#"*80 + "\nCOMPREHENSIVE MULTI-DISEASE BENCHMARK SYSTEM MATRIX\n" + "#"*80)
    for disease_context, report in engine.metrics_report.items():
        print(f"\n--- Diagnostic Outcome Matrix for {disease_context} ---")
        print(report.round(4).to_string(index=False))

from sklearn.metrics import accuracy_score, precision_score, confusion_matrix


# 1 = Disease Present, 0 = Healthy
y_actual =    [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]  # True patient health status
y_predicted = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]  

# 1. Compute Accuracy Score
# Formula: (True Positives + True Negatives) / Total Predictions
accuracy = accuracy_score(y_actual, y_predicted)

# 2. Compute Precision Score
# Formula: True Positives / (True Positives + False Positives)
precision = precision_score(y_actual, y_predicted, zero_division=0)

# 3. Compute Confusion Matrix
# Returns a 2x2 array: [[True Negatives, False Positives], [False Negatives, True Positives]]
matrix = confusion_matrix(y_actual, y_predicted)


print("========================================")
print("       MODEL EVALUATION METRICS         ")
print("========================================")
print(f"[+] Accuracy Score  : {accuracy:.4f} ({accuracy * 100:.1f}%)")
print(f"[+] Precision Score : {precision:.4f} ({precision * 100:.1f}%)")
print("\n[+] Confusion Matrix Array:")
print(matrix)


tn, fp, fn, tp = matrix.ravel()
print("\n--- Detailed Confusion Matrix Breakdown ---")
print(f"True Negatives  (Predicted Healthy, Actually Healthy) : {tn}")
print(f"False Positives (Predicted Disease, Actually Healthy) : {fp} <-- (Type I Error)")
print(f"False Negatives (Predicted Healthy, Actually Disease) : {fn} <-- (Type II Error - Critical in Medicine!)")
print(f"True Positives  (Predicted Disease, Actually Disease) : {tp}")
