# Credit Card Fraud Detection using Machine Learning

## Problem Statement
Credit card fraud is a highly imbalanced classification problem where fraudulent transactions represent a very small fraction of total activity.  
The goal of this project is to build a machine learning system that can accurately detect fraud while balancing precision and recall.

## Dataset
The dataset contains real-world inspired transaction data with:
- Transaction amount
- Time-based features
- Merchant category
- Geographic information
- Fraud label (`is_fraud`)

The dataset is highly imbalanced (~0.6% fraud cases).
## Project Workflow

### 1. Data Preprocessing
- Removed identifiers to prevent leakage
- Extracted time-based features (`hour`, `day_of_week`)
- Encoded categorical features using One-Hot Encoding
- Built preprocessing pipeline using `ColumnTransformer`

### 2. Baseline Model — Logistic Regression
- Used `class_weight='balanced'` to handle imbalance
- Achieved:
  - ROC-AUC ≈ 0.93
  - High recall but very low precision
- Showed limitations of linear models on non-linear fraud patterns

### 3. Exploratory Data Analysis
EDA revealed:
- Fraud transactions tend to have higher amounts
- Fraud patterns vary by hour and merchant category
- Relationships are non-linear → motivated tree-based models

### 4. Improved Model — XGBoost
- Used gradient boosting with imbalance handling (`scale_pos_weight`)
- Achieved:
  - ROC-AUC ≈ 0.995
  - Major improvement in fraud detection precision
- Captured non-linear interactions missed by logistic regression

### 5. Threshold Optimization
Instead of using default 0.5 threshold, evaluated precision–recall tradeoffs to simulate real-world fraud detection behavior.

## Key Learnings
- Fraud detection is not about accuracy — it’s about recall vs precision tradeoff.
- Linear models struggle with interaction-heavy tabular data.
- Tree-based boosting models (XGBoost) perform significantly better.
- Proper preprocessing pipelines prevent data leakage.
- Threshold tuning is critical for real-world deployment.

## Tech Stack
- Python
- Pandas / NumPy
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn

## Future Work
- Model explainability using SHAP
- Hyperparameter tuning
- Deployment as a fraud prediction API
