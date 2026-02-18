# 💳 Credit Card Fraud Detection System (Deployed ML Project)

## 📌 Project Overview

This project implements an end-to-end Machine Learning pipeline to detect fraudulent credit card transactions. The system handles severe class imbalance, identifies complex fraud patterns, and is deployed as a live web application using Streamlit.

The goal of this project is to build a practical fraud detection system that balances fraud detection accuracy with minimizing false alarms.

---

## 🚀 Live Demo

Access the deployed app here:  
👉 https://credit-card-fraud-detection-az123.streamlit.app

---

## 🧠 Problem Statement

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent a very small fraction of total transactions. Traditional accuracy metrics are misleading, making it necessary to focus on recall, precision, and threshold optimization.

---

## 📊 Dataset Description

The dataset contains anonymized credit card transaction data including:

- Transaction amount
- Merchant location
- Customer location
- Transaction category
- Time-based features
- Demographic attributes

Target variable:
- `0` → Normal Transaction  
- `1` → Fraudulent Transaction  

---

## ⚙️ Key Features of the Project

✔ End-to-end ML pipeline  
✔ Handling severe class imbalance  
✔ Threshold tuning for real-world decision making  
✔ Model comparison and evaluation  
✔ Live web deployment  
✔ Interactive prediction interface  

---

## 🛠️ Technologies Used

- Python
- Pandas & NumPy
- Scikit-Learn
- XGBoost
- Streamlit
- Joblib

---

## 🤖 Model Development Workflow

### 1️⃣ Data Preprocessing
- Feature engineering
- Handling categorical variables
- Scaling numerical features

---

### 2️⃣ Handling Class Imbalance
Used cost-sensitive learning with:
scale_pos_weight = (# Normal Transactions / # Fraud Transactions)


This ensured the model properly learned minority fraud patterns.

---

### 3️⃣ Baseline Model — Logistic Regression

Performance:
- Recall ≈ 61%
- Precision ≈ 27%

Limitations:
- Could not capture nonlinear fraud patterns.

---

### 4️⃣ Final Model — XGBoost

Why XGBoost?
- Captures complex nonlinear relationships
- Handles imbalanced data effectively
- Provides superior fraud detection capability

---

## 📈 Final Model Performance

After threshold optimization:

- Fraud Recall: **~85%**
- Fraud Precision: **~61%**
- ROC-AUC Score: **~0.99**

This demonstrates strong fraud detection ability while reducing false alarms.

---

## 🎯 Threshold Optimization Strategy

Instead of using default threshold (0.5), a custom threshold of **0.93** was selected to balance:

- Maximizing fraud detection
- Minimizing false positives

This reflects real-world fraud detection decision making.

---

## 🌐 Deployment

The trained ML pipeline was deployed as an interactive web application using Streamlit.

Features of the deployed app:
- User input interface
- Real-time fraud prediction
- Probability interpretation
- Clear decision output

---

## 📂 Project Structure
- Credit Card Fraud Project.ipynb
- Fraud Detection - XGBoost Model.ipynb
- app.py
- fraud_xgboost_pipeline.pkl
- requirements.txt
- README.md

---

## 🧩 Key Learnings

- Importance of handling imbalanced datasets
- Precision vs Recall trade-off in real systems
- Threshold tuning for business impact
- Building deployable ML pipelines
- Converting ML models into real-world applications

---

## 🔮 Future Improvements

- Real-time API integration
- Advanced feature engineering
- Model explainability (SHAP)
- Automated fraud alert system

---

⭐ If you found this project useful, consider giving it a star!
