#downloas stock data
import yfinance as yf
import pandas as pd
import numpy as np

# Download Tata Power stock data (Example ticker: TATAPOWER.NS for NSE)
df = yf.download("TATAPOWER.NS", start="2020-01-01", end="2025-02-07")
#print(df.dtypes)

df = pd.read_csv("TataPower_stock_data.csv")


data = df[['Close']].values

#define function to convert nd.array object into float data

def safe_to_float(val):
    try:
        return float(val)
    except ValueError:
        return np.nan
clean_data = np.vectorize(safe_to_float)(data)
clean_data = clean_data[~np.isnan(clean_data)]
clean_data=clean_data.reshape(-1,1)
print(clean_data.dtype)



#b. Preprocess the Data for LSTM Model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load Tata Power stock data
#df = pd.read_csv("TataPower_stock_data.csv")

# Use only 'Close' price
#data = df[['Close']].values

# Scale the data between 0 and 1
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(clean_data)

# Create training and testing datasets
train_size = int(len(clean_data) * 0.8)
train_data, test_data = data_scaled[:train_size], data_scaled[train_size:]

# Create features for prediction
def create_features(dataset, time_steps=30):
    X, Y = [], []
    for i in range(len(dataset) - time_steps):
        X.append(dataset[i:i + time_steps])
        Y.append(dataset[i + time_steps])
    return np.array(X), np.array(Y)

time_steps = 30
X_train, Y_train = create_features(train_data, time_steps)
X_test, Y_test = create_features(test_data, time_steps)

# Reshape for LSTM input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Check the shapes
print(f"Train Data Shape: {X_train.shape}")
print(f"Test Data Shape: {X_test.shape}") 


# Build and Train the LSTM Model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import MeanSquaredError
# Build LSTM Model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(time_steps, 1)),
    LSTM(50, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss=MeanSquaredError())

# Train the model
model.fit(X_train, Y_train, epochs=20, batch_size=16, validation_data=(X_test, Y_test))

# Save the model for later use
model.save("tata_power_lstm_model.h5")
