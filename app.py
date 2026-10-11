import streamlit as st
import pandas as pd
import numpy as np
import joblib
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")
# ---- Simple styling ----
st.markdown("""
    <style>
    .main { background-color: #f9fafb; }
    .price-box {
        background-color: #ecfdf5;
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 10px;
    }
    .price-box h2 { color: #065f46; margin: 0; }
    </style>
""", unsafe_allow_html=True)
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    metadata = joblib.load("metadata.pkl")
    return model, metadata
model, metadata = load_artifacts()    
LOG_TARGET = True
# Approximate INR -> NGN conversion. Update this periodically — exchange rates move.
INR_TO_NGN = 19.0  # change to today's actual rate before you commit
st.title("🚗 Car Price Predictor")
st.caption(f"Model: {metadata.get('model_name', 'Regression model')}")
st.write("Enter the car details below to estimate its selling price.")
with st.expander("ℹ️ About the currency shown"):
    st.write(
        "This model was trained on Indian (CarDekho) listings, so predictions are "
        "naturally in Indian Rupees. The Naira figure below is an approximate "
        "conversion for convenience, not a price calibrated to the Nigerian market — "
        "treat it as a rough reference, not a quote."
    )
col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Brand", metadata["brands"])
    year = st.number_input("Year of manufacture", min_value=1990, max_value=2026, value=2018)
    km_driven = st.number_input("Kilometres driven", min_value=0, max_value=500000, value=40000, step=1000)
    fuel = st.selectbox("Fuel type", metadata["fuel_options"])
    seller_type = st.selectbox("Seller type", metadata["seller_type_options"])
with col2:
    transmission = st.selectbox("Transmission", metadata["transmission_options"])
    owner = st.selectbox("Ownership", metadata["owner_options"])
    mileage = st.number_input("Mileage (km/ltr/kg)", min_value=0.0, max_value=42.0, value=18.0, step=0.1)
    engine = st.number_input("Engine (CC)", min_value=600, max_value=3600, value=1200, step=50)
    max_power = st.number_input("Max power (bhp)", min_value=0.0, max_value=400.0, value=80.0, step=1.0)
seats = st.slider("Seats", min_value=2, max_value=10, value=5)
currency = st.radio("Show price in:", ["₹ Indian Rupees (model's native output)", "₦ Naira (approx. converted)"], horizontal=True)
if st.button("Predict Price", type="primary"):
    car_age = 2026 - year
    input_df = pd.DataFrame([{
        "km_driven": km_driven,
        "mileage(km/ltr/kg)": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
        "car_age": car_age,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "brand": brand,
    }])
    raw_pred = model.predict(input_df)[0]
    pred_inr = np.expm1(raw_pred) if LOG_TARGET else raw_pred
    if currency.startswith("₦"):
        pred_display = pred_inr * INR_TO_NGN
        symbol = "₦"
    else:
        pred_display = pred_inr
        symbol = "₹"
    st.markdown(f"""
        <div class="price-box">
            <p style="margin:0; color:#047857;">Estimated Price</p>
            <h2>{symbol}{pred_display:,.0f}</h2>
        </div>
    """, unsafe_allow_html=True)
    st.caption(
        "This is an estimate from a model trained on CarDekho (India) listings. "
        "Actual resale value depends on condition, location, and market demand."
    )
