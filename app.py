import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="wide"
)

# ── Load model & scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load('best_model_classifier.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏦 Bank Churn Predictor")
st.markdown("Predict whether a customer will leave the bank based on their profile.")
st.divider()

# ── Sidebar — Client Input ────────────────────────────────────────────────────
st.sidebar.header("Client Profile")

credit_score      = st.sidebar.slider("Credit Score",       300, 850, 650)
age               = st.sidebar.slider("Age",                 18, 100,  35)
tenure            = st.sidebar.slider("Tenure (years)",       0,  10,   5)
balance           = st.sidebar.number_input("Balance (€)",   0.0, 300000.0, 50000.0, step=1000.0)
estimated_salary  = st.sidebar.number_input("Estimated Salary (€)", 0.0, 300000.0, 60000.0, step=1000.0)
num_of_products   = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4], index=1)
geography         = st.sidebar.selectbox("Country", ["France", "Germany", "Spain"])
gender            = st.sidebar.selectbox("Gender", ["Male", "Female"])
has_cr_card       = st.sidebar.radio("Has Credit Card?",    ["Yes", "No"], horizontal=True)
is_active_member  = st.sidebar.radio("Active Member?",      ["Yes", "No"], horizontal=True)

predict_btn = st.sidebar.button("Predict", type="primary", use_container_width=True)

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(credit_score, geography, gender, age, tenure,
               balance, num_of_products, has_cr_card, is_active_member, estimated_salary):
    is_active = 1 if is_active_member == "Yes" else 0
    data = {
        'CreditScore':     credit_score,
        'Gender':          1 if gender == "Male" else 0,  # label encoding: Female=0, Male=1
        'Age':             age,
        'Tenure':          tenure,
        'Balance':         balance,
        'NumOfProducts':   num_of_products,
        'HasCrCard':       1 if has_cr_card == "Yes" else 0,
        'IsActiveMember':  is_active,
        'EstimatedSalary': estimated_salary,
        'Geography':       geography
    }
    df = pd.DataFrame([data])
    df = pd.get_dummies(df, columns=['Geography'])
    # Feature engineering — must match training
    df['IsZeroBalance']        = (df['Balance'] == 0).astype(int)
    df['Balance_Salary_Ratio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['Age_Activity']         = df['Age'] * df['IsActiveMember']
    for col in scaler.feature_names_in_:
        if col not in df.columns:
            df[col] = 0
    df = df[scaler.feature_names_in_]
    return scaler.transform(df)

# ── Main layout ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Client Summary")
    summary = {
        "Credit Score": credit_score,
        "Age": age,
        "Country": geography,
        "Gender": gender,
        "Tenure": f"{tenure} years",
        "Balance": f"€{balance:,.0f}",
        "Estimated Salary": f"€{estimated_salary:,.0f}",
        "Products": num_of_products,
        "Credit Card": has_cr_card,
        "Active Member": is_active_member,
    }
    for k, v in summary.items():
        st.markdown(f"**{k}:** {v}")

with col2:
    if predict_btn:
        X = preprocess(credit_score, geography, gender, age, tenure,
                       balance, num_of_products, has_cr_card, is_active_member, estimated_salary)

        pred_class = model.predict(X)[0]
        pred_proba = model.predict_proba(X)[0]
        churn_prob = pred_proba[1]

        # ── Gauge chart ──────────────────────────────────────────────────────
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(churn_prob * 100, 1),
            title={"text": "Churn Risk (%)"},
            delta={"reference": 50},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#e74c3c" if churn_prob > 0.5 else "#2ecc71"},
                "steps": [
                    {"range": [0,  40], "color": "#d5f5e3"},
                    {"range": [40, 70], "color": "#fdebd0"},
                    {"range": [70, 100], "color": "#fadbd8"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": 50
                }
            }
        ))
        fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

        # ── Verdict ──────────────────────────────────────────────────────────
        if pred_class == 1:
            st.error(f"⚠️  **CHURN** — This client is likely to leave  ({churn_prob:.1%})")
        else:
            st.success(f"✅  **NO CHURN** — This client is likely to stay  ({1 - churn_prob:.1%})")

    else:
        st.info("Fill the client profile in the sidebar and click **Predict**.")

# ── Banking Overview ─────────────────────────────────────────────────────────
st.divider()
st.subheader("🏦 Banking Overview")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Industry Avg. Churn Rate", "20.4%", "-2.1% vs last year")
kpi2.metric("Active Member Rate", "51.5%", "+3.2% vs last year")
kpi3.metric("Avg. Customer Balance", "€76,486", "+€1,200 vs last year")
kpi4.metric("Avg. Tenure", "5.0 years", "+0.3 vs last year")

st.divider()

# ── Customer Profile vs Average Churner ──────────────────────────────────────
st.subheader("👤 Customer Profile vs. Average Churner")

avg_churner = {
    "Credit Score": 645,
    "Age":          44,
    "Tenure":        4,
    "Balance":   91108,
    "Products":      1,
    "Salary":    101465,
}

current = {
    "Credit Score": credit_score,
    "Age":          age,
    "Tenure":       tenure,
    "Balance":      int(balance),
    "Products":     num_of_products,
    "Salary":       int(estimated_salary),
}

radar_fig = go.Figure()
categories = list(avg_churner.keys())

def normalise(vals, keys):
    limits = {
        "Credit Score": (300, 850),
        "Age":          (18, 92),
        "Tenure":       (0, 10),
        "Balance":      (0, 250000),
        "Products":     (1, 4),
        "Salary":       (0, 200000),
    }
    return [round((vals[k] - limits[k][0]) / (limits[k][1] - limits[k][0]) * 100, 1) for k in keys]

radar_fig.add_trace(go.Scatterpolar(
    r=normalise(avg_churner, categories) + [normalise(avg_churner, categories)[0]],
    theta=categories + [categories[0]],
    fill='toself', name='Avg. Churner',
    line_color='#e74c3c', fillcolor='rgba(231,76,60,0.15)'
))
radar_fig.add_trace(go.Scatterpolar(
    r=normalise(current, categories) + [normalise(current, categories)[0]],
    theta=categories + [categories[0]],
    fill='toself', name='This Customer',
    line_color='#2ecc71', fillcolor='rgba(46,204,113,0.15)'
))
radar_fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
    showlegend=True, height=380,
    margin=dict(t=30, b=30, l=40, r=40)
)
st.plotly_chart(radar_fig, use_container_width=True)

# ── Churn Risk Factors ────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Key Churn Risk Factors")

f1, f2, f3 = st.columns(3)
with f1:
    st.markdown("""
    ** Geography**
    Germany customers churn at **32%** — nearly double France (16%) and Spain (17%).
    """)
with f2:
    st.markdown("""
    ** Number of Products**
    Customers with **3–4 products** have a churn rate above **80%**.
    Single-product customers churn at **27%**.
    """)
with f3:
    st.markdown("""
    ** Age**
    Customers aged **45–60** are the highest-risk segment.
    Younger customers (18–30) churn at under **10%**.
    """)
