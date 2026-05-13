import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("sales_model.pkl")

st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")

st.title("📊 Retail Sales Forecast Intelligence System")
st.write("Predict sales + simulate business decisions using ML")

# =========================
# USER INPUTS
# =========================
st.sidebar.header("🎛️ Input Controls")

store_nbr = st.sidebar.number_input("Store Number", min_value=1, max_value=54, value=1)
onpromotion = st.sidebar.number_input("Promotions", min_value=0, value=0)
month = st.sidebar.slider("Month", 1, 12, 1)
day = st.sidebar.slider("Day", 1, 31, 1)
weekday = st.sidebar.slider("Weekday", 0, 6, 0)
lag_1 = st.sidebar.number_input("Previous Day Sales", min_value=0.0, value=0.0)
rolling_mean_7 = st.sidebar.number_input("7-Day Rolling Mean", min_value=0.0, value=0.0)

is_weekend = 1 if weekday in [5, 6] else 0

# =========================
# BUILD INPUT DATA
# =========================
input_data = pd.DataFrame({
    "store_nbr": [store_nbr],
    "onpromotion": [onpromotion],
    "year": [2017],
    "month": [month],
    "day": [day],
    "weekday": [weekday],
    "is_weekend": [is_weekend],
    "lag_1": [lag_1],
    "rolling_mean_7": [rolling_mean_7]
})

# =========================
# ALIGN FEATURES
# =========================
model_features = model.feature_names_in_

for col in model_features:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[model_features]

# =========================
# PREDICTION SECTION
# =========================
st.subheader("🧮 Prediction Engine")

if st.button("Predict Sales"):

    prediction = model.predict(input_data)[0]

    st.metric(label="📦 Predicted Sales", value=f"{prediction:.2f}")

    # =========================
    # WHAT-IF SIMULATION
    # =========================
    st.subheader("🧪 What-If Simulation (Promotion Impact)")

    promo_test = st.slider("Simulate Promotion Change", 0, 1, onpromotion)

    input_data_test = input_data.copy()
    input_data_test["onpromotion"] = promo_test

    new_prediction = model.predict(input_data_test)[0]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Original Prediction", f"{prediction:.2f}")

    with col2:
        st.metric("Simulated Prediction", f"{new_prediction:.2f}")

    impact = new_prediction - prediction

    if impact > 0:
        st.success(f"📈 Promotion increases sales by {impact:.2f}")
    else:
        st.warning(f"📉 Promotion impact: {impact:.2f}")

# =========================
# QUICK INSIGHTS (OPTIONAL DATA VIEW)
# =========================
st.subheader("📊 Quick Insight Panel")

st.info("Use this section to extend into store/family analysis later.")

st.write("Model is trained and ready for multi-scenario forecasting.")