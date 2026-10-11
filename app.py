import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Styling (light theme) ----------------
st.markdown("""
    <style>
    .hero {
        padding: 28px 32px;
        border-radius: 16px;
        background-color: #eef2ff;
        border: 1px solid #c7d2fe;
        margin-bottom: 24px;
    }
    .hero h1 { margin: 0; font-size: 2rem; color: #1e293b; }
    .hero p { margin: 6px 0 0 0; color: #475569; }
    .price-card {
        background-color: #ecfdf5;
        border: 1px solid #10b981;
        border-radius: 18px;
        padding: 36px;
        text-align: center;
    }
    .price-card .label { color: #047857; font-size: 0.95rem; letter-spacing: 0.05em; text-transform: uppercase; }
    .price-card .amount { color: #065f46; font-size: 3rem; font-weight: 700; margin-top: 6px; }
    .price-card .note { color: #047857; font-size: 0.85rem; margin-top: 10px; }
    .placeholder-card {
        border: 1px dashed #cbd5e1;
        border-radius: 18px;
        padding: 48px;
        text-align: center;
        color: #64748b;
        background-color: #f8fafc;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    metadata = joblib.load("metadata.pkl")
    return model, metadata

model, metadata = load_artifacts()

LOG_TARGET = True
INR_TO_NGN = 19.0  # update to today's rate periodically

# ---------------- Hero header ----------------
st.markdown(f"""
    <div class="hero">
        <h1>🚗 Car Price Predictor</h1>
        <p>Model: {metadata.get('model_name', 'Regression model')} · Trained on CarDekho (India) listings</p>
    </div>
""", unsafe_allow_html=True)

# ---------------- Sidebar: all inputs ----------------
with st.sidebar:
    st.header("Car Details")

    brand = st.selectbox("Brand", metadata["brands"])
    year = st.number_input("Year of manufacture", min_value=1990, max_value=2026, value=2018)
    km_driven = st.number_input("Kilometres driven", min_value=0, max_value=500000, value=40000, step=1000)

    st.divider()

    fuel = st.selectbox("Fuel type", metadata["fuel_options"])
    transmission = st.selectbox("Transmission", metadata["transmission_options"])
    seller_type = st.selectbox("Seller type", metadata["seller_type_options"])
    owner = st.selectbox("Ownership", metadata["owner_options"])

    st.divider()

    mileage = st.number_input("Mileage (km/ltr/kg)", min_value=0.0, max_value=42.0, value=18.0, step=0.1)
    engine = st.number_input("Engine (CC)", min_value=600, max_value=3600, value=1200, step=50)
    max_power = st.number_input("Max power (bhp)", min_value=0.0, max_value=400.0, value=80.0, step=1.0)
    seats = st.slider("Seats", min_value=2, max_value=10, value=5)

    st.divider()

    currency = st.radio(
        "Show price in",
        ["\u20b9 INR (native)", "\u20a6 NGN (approx.)"],
    )

    predict_clicked = st.button("Predict Price", type="primary", use_container_width=True)

    with st.expander("ℹ️ About the currency"):
        st.caption(
            "Trained on Indian listings, so predictions are naturally in Rupees. "
            "The Naira figure is an approximate conversion for convenience only, "
            "not a price calibrated to the Nigerian market."
        )

# ---------------- Main area: result ----------------
left, right = st.columns([3, 2])

with left:
    if predict_clicked:
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

        if currency.startswith("\u20a6"):
            pred_display = pred_inr * INR_TO_NGN
            symbol = "\u20a6"
        else:
            pred_display = pred_inr
            symbol = "\u20b9"

        st.markdown(f"""
            <div class="price-card">
                <div class="label">Estimated Selling Price</div>
                <div class="amount">{symbol}{pred_display:,.0f}</div>
                <div class="note">{brand} · {year} · {transmission}</div>
            </div>
        """, unsafe_allow_html=True)

        st.caption(
            "This is an estimate from a model trained on CarDekho (India) listings. "
            "Actual resale value depends on condition, location, and market demand."
        )
    else:
        st.markdown("""
            <div class="placeholder-card">
                Fill in the car details in the sidebar, then click<br><b>Predict Price</b> to see the estimate here.
            </div>
        """, unsafe_allow_html=True)

with right:
    st.subheader("Quick summary")
    c1, c2 = st.columns(2)
    c1.metric("Brand", brand)
    c2.metric("Year", year)
    c1.metric("Km driven", f"{km_driven:,}")
    c2.metric("Fuel", fuel)
    c1.metric("Transmission", transmission)
    c2.metric("Owner", owner)
