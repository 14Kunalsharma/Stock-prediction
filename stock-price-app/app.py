import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Moved to top for reliability
import yfinance as yf
import streamlit as st

st.set_page_config(page_title="Stock Predictor", layout="centered")

st.title("📈 Stock Price Prediction App")

# Load model and scaler
try:
    model = joblib.load("stock-price-app/model.pkl")
    scaler = joblib.load("stock-price-app/scaler.pkl")
    st.success("Model and Scaler loaded successfully ✅")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

st.write("Enter stock parameters to predict closing price")

# Inputs
open_price = st.number_input("Open Price", value=150.0)
high_price = st.number_input("High Price", value=155.0)
low_price = st.number_input("Low Price", value=148.0)
volume = st.number_input("Volume", value=1000000.0)

# Prediction
if st.button("Predict"):
    try:
        # 1. Feature Engineering: Create the 2 missing features found in your scaler 
        price_range = high_price - low_price
        avg_price = (high_price + low_price) / 2

        # 2. Build the input array with all 6 features in the correct order 
        # Order: Open, High, Low, Volume, Price_Range, Avg_Price
        data = np.array([[open_price, high_price, low_price, volume, price_range, avg_price]])
        
        # 3. Transform and Predict
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)

        st.metric("Predicted Closing Price", f"${prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Chart
try:
    df = yf.download("TSLA", start="2020-01-01")
    st.subheader("Historical Trend")
    st.line_chart(df["Close"])
except:
    st.info("Note: Tesla.csv not found. Upload it to see historical charts.")
