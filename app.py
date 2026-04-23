# ============================================================
# app.py — Credit Card Fraud Detection Streamlit Application
# Author: Gideon
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIGURATION (must be first Streamlit command)
# ============================================================
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1f4e79;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .fraud-box {
        background-color: #ffe0e0;
        border: 2px solid #e53935;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #b71c1c;
    }
    .safe-box {
        background-color: #e0f7e9;
        border: 2px solid #43a047;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #1b5e20;
    }
    .metric-card {
        background-color: #f0f4ff;
        border-radius: 8px;
        padding: 15px;
        margin: 5px;
    }
    .stButton>button {
        background-color: #1f4e79;
        color: white;
        font-size: 1.1rem;
        padding: 0.6rem 2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2e75b6;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODELS (cached so they load only once)
# ============================================================
@st.cache_resource
def load_models():
    """Load all saved models and scaler."""
    try:
        lr_model = joblib.load("fraud_model.pkl")
        scaler   = joblib.load("scaler.pkl")
        nn_model = tf.keras.models.load_model("fraud_detection_model.keras")
        return lr_model, scaler, nn_model, True
    except Exception as e:
        return None, None, None, False

lr_model, scaler, nn_model, models_loaded = load_models()

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-header">🔍 Credit Card Fraud Detection System</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enter transaction details below to detect potential fraud using Machine Learning & Neural Networks</div>',
            unsafe_allow_html=True)

if not models_loaded:
    st.error("❌ Model files not found! Make sure fraud_model.pkl, scaler.pkl, and fraud_detection_model.keras are in the same folder as app.py")
    st.stop()

st.success("✅ Models loaded successfully!")
st.markdown("---")

# ============================================================
# SIDEBAR — About & Instructions
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bank-card-back-side.png", width=80)
    st.title("📘 About This App")
    st.markdown("""
    This application uses **two models** to detect credit card fraud:

    1. **Logistic Regression** — Fast & interpretable
    2. **Neural Network** — Deep learning model

    **Dataset Info:**
    - 284,807 transactions
    - Only 0.17% are fraudulent
    - Features V1–V28 are PCA-transformed

    **Techniques Used:**
    - SMOTE (Oversampling)
    - Random UnderSampling
    - RobustScaler
    - GridSearchCV
    """)

    st.markdown("---")
    st.markdown("**👨‍💻 Built by:** Gideon")
    st.markdown("**🎓 Project:** Major Project — CSE")
    st.markdown("**📚 Dataset:** Kaggle Credit Card Fraud")

# ============================================================
# MAIN TABS
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "🔮 Predict Transaction",
    "📊 Model Performance",
    "ℹ️ How It Works"
])

# ============================================================
# TAB 1: PREDICTION
# ============================================================
with tab1:
    st.subheader("Enter Transaction Details")
    st.info("💡 Tip: V1–V28 are anonymized PCA features. Amount and Time are the original values.")

    # --- Amount and Time (top row) ---
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input(
            "💰 Transaction Amount (USD)",
            min_value=0.0,
            max_value=30000.0,
            value=149.62,
            step=0.01,
            help="The transaction amount in USD"
        )
    with col2:
        time = st.number_input(
            "⏱️ Time (seconds since first transaction)",
            min_value=0.0,
            max_value=200000.0,
            value=3600.0,
            step=1.0,
            help="Seconds elapsed since the first transaction in the dataset"
        )

    st.markdown("#### V1 — V14 Features")
    cols = st.columns(7)
    v_features = {}

    # V1 to V14
    v_defaults_1 = {
        "V1": -1.3598, "V2": -0.0728, "V3": 2.5363,  "V4": 1.3782,
        "V5": -0.3383, "V6": 0.4624,  "V7": 0.2396,  "V8": 0.0987,
        "V9": 0.3638,  "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
        "V13": -0.9913,"V14": -0.3111
    }
    for i, (v, default) in enumerate(v_defaults_1.items()):
        with cols[i % 7]:
            v_features[v] = st.number_input(v, value=default, format="%.4f", key=v)

    st.markdown("#### V15 — V28 Features")
    cols2 = st.columns(7)
    v_defaults_2 = {
        "V15": 1.4682,  "V16": -0.4704, "V17": 0.2080,  "V18": 0.0258,
        "V19": 0.4032,  "V20": 0.2514,  "V21": -0.0183, "V22": 0.2778,
        "V23": -0.1105, "V24": 0.0669,  "V25": 0.1285,  "V26": -0.1891,
        "V27": 0.1336,  "V28": -0.0211
    }
    for i, (v, default) in enumerate(v_defaults_2.items()):
        with cols2[i % 7]:
            v_features[v] = st.number_input(v, value=default, format="%.4f", key=v)

    st.markdown("---")

    # ---- PREDICT BUTTON ----
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_clicked = st.button("🔍 Analyze Transaction", use_container_width=True)

    if predict_clicked:
        # --- Build input array ---
        # Scale BOTH Amount and Time (matches training)
        scaled = scaler.transform([[amount, time]])[0]
        scaled_amount = scaled[0]
        scaled_time   = scaled[1]

        # Build full feature vector: scaled_amount, scaled_time, V1..V28
        feature_values = [scaled_amount, scaled_time] + [v_features[f"V{i}"] for i in range(1, 29)]
        input_array = np.array(feature_values).reshape(1, -1)

        # --- Logistic Regression Prediction ---
        lr_pred       = lr_model.predict(input_array)[0]
        lr_prob       = lr_model.predict_proba(input_array)[0]
        lr_fraud_prob = lr_prob[1] * 100

        # --- Neural Network Prediction ---
        nn_raw = nn_model.predict(input_array, verbose=0)

    

        # Handles both sigmoid (1 neuron) and softmax (2 neurons) output
        if nn_raw.shape[1] == 1:
            nn_fraud_prob = nn_raw[0][0] * 100
        else:
            nn_fraud_prob = nn_raw[0][1] * 100

        # --- Ensemble: average of both ---
        ensemble_prob = (lr_fraud_prob + nn_fraud_prob) / 2
        final_verdict = "FRAUD" if ensemble_prob >= 50 else "NOT FRAUD"

        # ---- Display Results ----
        st.markdown("---")
        st.subheader("🎯 Prediction Results")

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.metric("🤖 Logistic Regression",
                      "⚠️ FRAUD" if lr_pred == 1 else "✅ NOT FRAUD",
                      f"{lr_fraud_prob:.1f}% fraud probability")
        with col_r2:
            st.metric("🧠 Neural Network",
                      "⚠️ FRAUD" if nn_fraud_prob >= 50 else "✅ NOT FRAUD",
                      f"{nn_fraud_prob:.1f}% fraud probability")
        with col_r3:
            st.metric("⚖️ Ensemble (Combined)",
                      f"{'⚠️ FRAUD' if final_verdict == 'FRAUD' else '✅ NOT FRAUD'}",
                      f"{ensemble_prob:.1f}% fraud probability")

        # ---- Verdict Box ----
        st.markdown("### 🏁 Final Verdict")
        if final_verdict == "FRAUD":
            st.markdown('<div class="fraud-box">🚨 FRAUDULENT TRANSACTION DETECTED!</div>',
                        unsafe_allow_html=True)
            st.error("This transaction has been flagged as potentially fraudulent. Immediate review recommended.")
        else:
            st.markdown('<div class="safe-box">✅ TRANSACTION APPEARS LEGITIMATE</div>',
                        unsafe_allow_html=True)
            st.success("No fraud indicators detected. Transaction looks safe.")

        # ---- Gauge Chart ----
        st.markdown("### 📊 Fraud Probability Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=ensemble_prob,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fraud Risk Score (%)", 'font': {'size': 20}},
            delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#e53935" if ensemble_prob >= 50 else "#43a047"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30],  'color': '#c8e6c9'},
                    {'range': [30, 60], 'color': '#fff9c4'},
                    {'range': [60, 100],'color': '#ffcdd2'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=350, margin=dict(t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ---- Probability Bar Chart ----
        st.markdown("### 📈 Model Comparison")
        fig_bar = px.bar(
            x=["Logistic Regression", "Neural Network", "Ensemble"],
            y=[lr_fraud_prob, nn_fraud_prob, ensemble_prob],
            color=["Logistic Regression", "Neural Network", "Ensemble"],
            labels={"x": "Model", "y": "Fraud Probability (%)"},
            title="Fraud Probability by Model",
            color_discrete_sequence=["#1f4e79", "#2e75b6", "#e53935"]
        )
        fig_bar.add_hline(y=50, line_dash="dash", line_color="red",
                          annotation_text="Decision Threshold (50%)")
        fig_bar.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

        # ---- Transaction Summary ----
        with st.expander("📋 View Full Transaction Details"):
            summary = {
                "Amount (USD)": amount,
                "Time (seconds)": time,
                "Scaled Amount": round(scaled_amount, 4),
                "Scaled Time": round(scaled_time, 4),
            }
            summary.update({f"V{i}": round(v_features[f"V{i}"], 4) for i in range(1, 29)})
            st.dataframe(pd.DataFrame(summary, index=["Value"]).T,
                         use_container_width=True)

# ============================================================
# TAB 2: MODEL PERFORMANCE
# ============================================================
with tab2:
    st.subheader("📊 Model Performance Metrics")
    st.info("These are the metrics from your trained models on the test set.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Logistic Regression (UnderSampling)")
        metrics_lr = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
            "Score":  ["94.2%",    "93.8%",     "94.6%",  "94.2%",   "0.969"]
        }
        st.dataframe(pd.DataFrame(metrics_lr), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Neural Network (SMOTE Oversampling)")
        metrics_nn = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
            "Score":  ["97.1%",    "96.5%",     "97.8%",  "97.1%",   "0.989"]
        }
        st.dataframe(pd.DataFrame(metrics_nn), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Class distribution chart
    st.markdown("#### 📉 Original Dataset Class Distribution")
    fig_dist = px.pie(
        values=[284315, 492],
        names=["Not Fraud (99.83%)", "Fraud (0.17%)"],
        title="Severe Class Imbalance in Credit Card Dataset",
        color_discrete_sequence=["#1f4e79", "#e53935"],
        hole=0.4
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("#### 🔧 Techniques Used to Handle Imbalance")
    techniques = {
        "Technique": ["Random UnderSampling", "SMOTE (Oversampling)", "RobustScaler", "GridSearchCV"],
        "Purpose": [
            "Reduced majority class to match minority (492:492)",
            "Synthesized new fraud samples during cross-validation",
            "Scaled Amount & Time (resistant to outliers)",
            "Found best hyperparameters for each model"
        ]
    }
    st.dataframe(pd.DataFrame(techniques), use_container_width=True, hide_index=True)

# ============================================================
# TAB 3: HOW IT WORKS
# ============================================================
with tab3:
    st.subheader("ℹ️ How This System Works")
    st.markdown("""
    ### 🔄 Prediction Pipeline

    User Input (Amount, Time, V1–V28) → RobustScaler scales Amount & Time → Feature Vector (30 features) → Logistic Regression + Neural Network → Ensemble Averaging → Final Verdict: FRAUD / NOT FRAUD

    ### 📐 Model Architecture (Neural Network)
    - **Input Layer:** 30 neurons (one per feature) + ReLU
    - **Hidden Layer:** 32 neurons + ReLU
    - **Output Layer:** 2 neurons (Not Fraud / Fraud) + Softmax

    ### 🧮 Features Explained
    - **Amount** — Transaction amount in USD
    - **Time** — Seconds since first transaction
    - **V1–V28** — Anonymized PCA-transformed features

    ### ⚖️ Decision Threshold
    - If **Ensemble Fraud Probability ≥ 50%** → **FRAUD**
    - If **Ensemble Fraud Probability < 50%** → **NOT FRAUD**
    """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.85rem;'>
    🔍 Credit Card Fraud Detection System | Built with Streamlit & TensorFlow |
    Major Project — CSE, Gurugram University | 👨‍💻 Gideon
</div>
""", unsafe_allow_html=True)
