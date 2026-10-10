import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Set to True ONLY if you trained the model on log(selling_price)
LOG_TARGET = False

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")


@st.cache_resource
def load_files():
    model = joblib.load("model.pkl")
    meta = joblib.load("metadata.pkl")
    return model, meta


model, meta = load_files()

st.title("🚗 Used Car Price Predictor")
st.write("Enter the car details below to estimate its selling price.")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", meta["brands"])
    fuel = st.selectbox("Fuel type", meta["fuel_options"])
    transmission = st.selectbox("Transmission", meta["transmission_options"])
    seller_type = st.selectbox("Seller type", meta["seller_type_options"])
    owner = st.selectbox("Owner", meta["owner_options"])

with col2:
    km_driven = st.number_input("Kilometres driven", min_value=1, max_value=475000, value=50000, step=1000)
    car_age = st.number_input("Car age (years)", min_value=3, max_value=29, value=5, step=1)
    mileage = st.number_input("Mileage (km/ltr/kg)", min_value=9.0, max_value=42.0, value=18.0, step=0.5)
    engine = st.number_input("Engine (CC)", min_value=624.0, max_value=3604.0, value=1200.0, step=50.0)
    max_power = st.number_input("Max power (bhp)", min_value=32.8, max_value=400.0, value=85.0, step=1.0)
    seats = st.number_input("Seats", min_value=4, max_value=14, value=5, step=1)

if st.button("Predict price", type="primary"):
    row = pd.DataFrame([{
        "km_driven": km_driven,
        "car_age": car_age,
        "mileage(km/ltr/kg)": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": float(seats),
        "brand": brand,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
    }])

    # Make sure columns are in the exact order the model was trained on
    row = row[list(model.feature_names_in_)]

    pred = float(model.predict(row)[0])
    if LOG_TARGET:
        pred = float(np.expm1(pred))

    st.success(f"Estimated price: ₹{pred:,.0f}")
    st.caption(f"Model: {meta.get('model_name', 'N/A')}")