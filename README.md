# 🏦 Bank Customer Churn Prediction  End-to-End ML Pipeline

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/yahyahafid/bank-customer-churn-prediction-end-to-end-ml-pip)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Yahya%20Hafid-blue?logo=linkedin)](https://www.linkedin.com/in/yahya-h-b21805196/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> A production-ready machine learning pipeline to predict which bank customers are likely to churn — enabling proactive retention and reducing revenue loss.

---

## 📌 Business Problem

Customer churn costs banks millions annually. Acquiring a new customer is **5–7x more expensive** than retaining an existing one. This project identifies at-risk customers **before they leave**, giving the bank a window to act.

---

## 📂 Project Structure

```
bank-churn-prediction/
│
├── Analysis.ipynb               ← Full ML pipeline notebook
├── app.py                       ← Streamlit web app
├── best_model_classifier.pkl    ← Saved LightGBM model
├── scaler.pkl                   ← Saved StandardScaler
├── requirements.txt             ← All dependencies
└── README.md
```

---

## 🔁 Pipeline Overview

| Step | Description |
|------|-------------|
| 1 | **Data Understanding** — Load, inspect, and describe 10,000 customers |
| 2 | **EDA** — Distributions, correlations, churn by geography/gender/age |
| 3 | **Feature Engineering** — New behavioral variables from raw features |
| 4 | **Preprocessing** — Label Encoding, Standard Scaling, SMOTE balancing |
| 5 | **Model Training** — Logistic Regression, KNN, LightGBM |
| 6 | **Evaluation** — Accuracy, F1-Score, ROC-AUC, Confusion Matrix |
| 7 | **Hyperparameter Tuning** — Optimized LightGBM |
| 8 | **Model Export** — Saved with joblib |
| 9 | **Deployment** — Streamlit web app with churn risk gauge |

---

## 🏆 Results

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Logistic Regression | Baseline | — |
| K-Nearest Neighbors | — | — |
| **LightGBM** ✅ | **Best** | **Best** |

- Class imbalance (~20% churn) handled with **SMOTE**
- Final model saved as `best_model_classifier.pkl`

---

## 💡 Key Insights

- 📍 **German customers** churn at 2× the rate of French and Spanish customers
- 👴 **Older customers (45–60)** are significantly more likely to leave
- 📦 Customers with **only 1 product** churn far more than those with 2+
- 💤 **Inactive members** churn regardless of their account balance

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/-Scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![LightGBM](https://img.shields.io/badge/-LightGBM-02569B)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Seaborn](https://img.shields.io/badge/-Seaborn-3776AB)

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/yahyahafid/bank-churn-prediction
cd bank-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit app
streamlit run app.py
```

App opens at **http://localhost:8501**

---

## 📓 Full Notebook

View the complete analysis on Kaggle:
👉 [Bank Customer Churn Prediction — Kaggle Notebook](https://www.kaggle.com/code/yahyahafid/bank-customer-churn-prediction-end-to-end-ml-pip)

---

## 👤 Author

**Yahya Hafid**
- 🔗 [LinkedIn](https://www.linkedin.com/in/yahya-h-b21805196/)
- 🔗 [Kaggle](https://www.kaggle.com/yahyahafid)

---

## 📄 License

This project is licensed under the MIT License.
