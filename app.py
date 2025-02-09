import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the trained LSTM model
model = tf.keras.models.load_model("tata_power_lstm_model.h5")

# Streamlit UI
st.title("📉 Tata Power Stock Price Prediction App")
ticker = st.text_input("Enter Stock Symbol", "TATAPOWER.NS")

if st.button("Predict"):
    # Download stock data
    df = yf.download(ticker, start="2010-01-01", end="2025-02-07")
    
    if not df.empty:
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(df[['Close']].values)

        # Create input for prediction
        time_steps = 30
        last_data = data_scaled[-time_steps:].reshape(1, time_steps, 1)
        
        # Predict Next Day Price
        predicted_price = model.predict(last_data)
        predicted_price = scaler.inverse_transform(predicted_price)[0][0]

        # Show Result
        st.subheader(f"📊 Predicted Closing Price for Next Day: {predicted_price:.2f} INR")
        
        # Plot Closing Price
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df['Close'], label="Actual Price", color='blue')
        ax.axhline(predicted_price, color='red', linestyle='--', label="Predicted Price")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Invalid Stock Symbol. Try Again.")
