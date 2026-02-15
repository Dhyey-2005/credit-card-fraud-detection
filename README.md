# Credit Card Fraud Detection using Machine Learning

## Project Overview

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. Fraud detection is a highly imbalanced classification problem where fraudulent cases are extremely rare compared to normal transactions.

The objective of this project was to build an effective fraud detection model that can maximize fraud detection (recall) while minimizing false alarms (precision).

---

## Dataset Description

The dataset contains anonymized credit card transaction records with behavioral features and a binary target variable:

- **0 → Normal Transaction**
- **1 → Fraudulent Transaction**

Key challenges:

- Highly imbalanced dataset
- Fraud patterns are complex and nonlinear
- High cost of misclassification

---

## Project Workflow

The project was completed following a structured machine learning pipeline:

1. Data Cleaning and Preprocessing  
2. Exploratory Data Analysis (EDA)  
3. Handling Class Imbalance  
4. Baseline Model – Logistic Regression  
5. Threshold Tuning  
6. Advanced Model – XGBoost  
7. Model Evaluation and Comparison  
8. External Dataset Validation  

---

## Models Used

###  Logistic Regression (Baseline Model)

- Applied class weighting to handle imbalance
- Performed threshold tuning to improve precision
- Served as baseline model

Performance (after tuning):

- Recall ≈ 61%
- Precision ≈ 27%

---

### XGBoost (Final Model)

- Used `scale_pos_weight` to handle class imbalance
- Captured nonlinear fraud patterns effectively
- Achieved significantly better performance

Final performance (after threshold tuning):

- Recall ≈ 85%
- Precision ≈ 61%
- ROC-AUC ≈ 0.99

---

## Model Evaluation Strategy

Due to class imbalance, accuracy alone is not meaningful.

Key evaluation metrics used:

- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

Threshold tuning was performed to balance fraud detection capability and false alarm rate.

---

## External Validation

The final model was tested on a completely unseen dataset to ensure real-world reliability.

External test performance:

- Recall ≈ 79%
- Precision ≈ 50%

This confirms that the model generalizes well and does not suffer from overfitting.

---

## Model Saving

The final trained pipeline (preprocessing + XGBoost model) was saved using Joblib for deployment readiness.

---

## Key Learnings

- Handling class imbalance is critical in fraud detection
- Threshold tuning is essential for real-world ML systems
- Linear models have limitations in complex pattern detection
- Ensemble models like XGBoost significantly improve performance
- Model evaluation should always include external validation

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-Learn
- XGBoost
- Matplotlib, Seaborn
- Joblib

---

## Project Structure
- Logistic Regression Notebook
- XGBoost Model Notebook
- Saved Model Pipeline (.pkl)
- README.md

## Future Improvements
- Real-time fraud detection pipeline
- Feature engineering using transaction history
- Hyperparameter tuning using Optuna
- Model deployment via API or web application
