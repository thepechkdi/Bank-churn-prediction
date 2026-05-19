# 🏦 Bank Customer Churn Prediction  End-to-End ML Pipeline

<div align="center">

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/yahyahafid/bank-customer-churn-prediction-end-to-end-ml-pip)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**A production-ready machine learning pipeline to predict which bank customers are likely to churn - enabling proactive retention strategies and reducing revenue loss.**

</div>

----

## Abstract

Customer churn remains one of the most costly challenges in retail banking. Acquiring a new customer costs **5–7× more** than retaining an existing one, making early identification of at-risk customers a high-value business objective. This project delivers a rigorous, end-to-end machine learning solution built on 10,000 real customer records. Using exploratory data analysis, feature engineering, SMOTE-based class balancing, and gradient boosting, the pipeline identifies the key demographic and behavioral drivers of churn — geography, age, product usage, and membership activity — and produces a deployable LightGBM classifier with an interactive Streamlit web application for real-time prediction.

----

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Methodology](#methodology)
4. [Dataset](#dataset)
5. [Requirements & Installation](#requirements--installation)
6. [How to Run](#how-to-run)
7. [Key Results](#key-results)
8. [Outputs & Figures](#outputs--figures)
9. [License](#license)
10. [Author](#author)

----

## Project Overview

| Attribute       | Detail                                                              |
|-----------------|---------------------------------------------------------------------|
| **Domain**      | Machine Learning / Customer Analytics                               |
| **Industry**    | Retail Banking                                                      |
| **Dataset**     | Churn Modelling — 10,000 customers, 14 features                     |
| **Target**      | `Exited` (1 = churned, 0 = stayed)                                  |
| **Methods**     | EDA, Feature Engineering, SMOTE, Classification, Hyperparameter Tuning |
| **Best Model**  | LightGBM                                                            |
| **Tools**       | Python, Scikit-learn, LightGBM, imbalanced-learn, Streamlit         |
| **Notebook**    | [View on Kaggle](https://www.kaggle.com/code/yahyahafid/bank-customer-churn-prediction-end-to-end-ml-pip) |

----

## Project Structure

```
bank-churn-prediction/
│
├── 📂 notebooks/
│   └── Analysis.ipynb                  # Full end-to-end ML pipeline
│
├── 📂 models/
│   ├── best_model_classifier.pkl       # Saved LightGBM model
│   └── scaler.pkl                      # Saved StandardScaler
│
│
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

----

## Methodology

The pipeline is structured into nine analytical and engineering modules:

| Module | Description |
|--------|-------------|
| **1. Data Understanding** | Load, inspect, and statistically describe 10,000 customer records |
| **2. Exploratory Data Analysis** | Univariate, bivariate, and correlation analysis; churn patterns by geography, gender, age, balance |
| **3. Feature Engineering** | Construct new behavioral variables from raw features to improve signal |
| **4. Preprocessing** | Label Encoding for categorical variables; Standard Scaling for numerical features |
| **5. Class Balancing** | SMOTE oversampling to correct the 80/20 churn imbalance |
| **6. Model Training** | Comparative training of Logistic Regression, K-Nearest Neighbors, and LightGBM |
| **7. Evaluation** | Accuracy, F1-Score, ROC-AUC, Confusion Matrix, Classification Report |
| **8. Hyperparameter Tuning** | Grid/random search to optimize LightGBM performance |
| **9. Deployment** | joblib model export + Streamlit app with real-time churn risk gauge |

----

## Dataset

| Attribute       | Detail |
|-----------------|--------|
| **Source**      | [Kaggle — Churn Modelling](https://www.kaggle.com/datasets/yahyahafid/churn-modelling) |
| **Records**     | 10,000 customers |
| **Features**    | 14 (demographic, financial, behavioral) |
| **Target**      | `Exited` — binary churn label |
| **Churn Rate**  | ~20% (class imbalance handled with SMOTE) |

### Feature Descriptions

| Feature | Type | Role |
|---------|------|------|
| `CreditScore` | Numerical | Explanatory |
| `Geography` | Categorical | Explanatory |
| `Gender` | Categorical | Explanatory |
| `Age` | Numerical | Explanatory |
| `Tenure` | Numerical | Explanatory |
| `Balance` | Numerical | Explanatory |
| `NumOfProducts` | Numerical | Explanatory |
| `HasCrCard` | Binary | Explanatory |
| `IsActiveMember` | Binary | Explanatory |
| `EstimatedSalary` | Numerical | Explanatory |
| `Exited` | Binary | **Target** |

----

## Requirements & Installation

### Prerequisites

- Python **3.11+**
- `pip`
- Jupyter Notebook or JupyterLab

### Install Dependencies

```bash
git clone https://github.com/yahyahafid/bank-churn-prediction.git
cd bank-churn-prediction
pip install -r requirements.txt
```

### Core Libraries

| Library | Purpose |
|---------|---------|
| `pandas` | Data wrangling and analysis |
| `numpy` | Numerical computation |
| `matplotlib` | Static visualisation |
| `seaborn` | Statistical plots |
| `scikit-learn` | Preprocessing, modeling, evaluation |
| `lightgbm` | Gradient boosting classifier |
| `imbalanced-learn` | SMOTE class balancing |
| `streamlit` | Web application deployment |
| `plotly` | Interactive charts in the app |
| `joblib` | Model serialization |

----

## How to Run

### Option A : Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/yahyahafid/bank-churn-prediction.git
cd bank-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook notebooks/Analysis.ipynb

# 4. Launch the Streamlit app
streamlit run app.py
```

App opens at **http://localhost:8501**

### Option B : Run on Kaggle (No Setup Required)

Click the badge below to open the full notebook in Kaggle — zero installation needed:

[![Open in Kaggle](https://img.shields.io/badge/Open%20in-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/code/yahyahafid/bank-customer-churn-prediction-end-to-end-ml-pip)

### Option C : Run on Google Colab

```
https://colab.research.google.com/github/yahyahafid/bank-churn-prediction/blob/main/notebooks/Analysis.ipynb
```

----

## Key Results

### Model Comparison

| Model | Performance |
|-------|-------------|
| Logistic Regression | Baseline |
| K-Nearest Neighbors | Moderate |
| **LightGBM** ✅ | **Best Highest F1 & ROC-AUC** |

### Key Insights from EDA

-  **German customers** churn at 2× the rate of French and Spanish customers
-  **Customers aged 45–60** show significantly higher churn probability
-  Customers with **only 1 product** churn far more than those with 2+
-  **Inactive members** churn regardless of their account balance or salary
-  High-balance customers who are inactive represent the highest-risk segment

----

## Outputs & Figures

All figures are saved to `/figures/` and models to `/models/` after a full notebook run.

Key visualisations produced:

- **Churn Distribution** : overall and by key features
- **Correlation Heatmap** : feature relationships
- **Age & Balance vs Churn** : distribution comparison
- **ROC Curve** :  model comparison across all classifiers
- **Confusion Matrix** : LightGBM final evaluation
- **Feature Importance Chart** : top drivers of churn

----

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License — Copyright (c) 2026 Yahya Hafid
```

Free to use, modify, and distribute with attribution.

---

## Author

<div align="center">

**Yahya Hafid**  
*Data Scientist & AI/ML Engineer*  
Tangier, Morocco 🇲🇦

[![Kaggle](https://img.shields.io/badge/Kaggle-yahyahafid-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/yahyahafid)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Yahya%20Hafid-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yahya-h-b21805196/)

*If you found this project useful, consider leaving a ⭐ on the repo and an upvote on the Kaggle notebook!*

</div>

---

<div align="center">
<sub>Built with 🧠 Machine Learning · 📊 Data · 🏦 Banking Analytics</sub>
</div>
