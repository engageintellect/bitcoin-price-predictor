import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from ta.momentum import RSIIndicator
from ta.trend import MACD
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import json
import os

def fetch_bitcoin_data():
    """Fetch Bitcoin historical data using yfinance with a longer history."""
    print("Fetching Bitcoin data...")
    btc_data = yf.download('BTC-USD', start='2010-01-01', end=pd.Timestamp.now().strftime('%Y-%m-%d'))
    print(f"Data fetched: {len(btc_data)} records")
    return btc_data

def add_technical_indicators(data):
    """Add moving averages, volume change, RSI, and MACD to the dataset."""
    data['Price_Change'] = data['Close'].pct_change()
    data['Price_Up'] = np.where(data['Price_Change'] > 0, 1, 0)
    
    # Moving Averages
    data['MA7'] = data['Close'].rolling(window=7).mean()
    data['MA21'] = data['Close'].rolling(window=21).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()

    # Volume Change
    data['Volume_Change'] = data['Volume'].pct_change()

    # RSI (Relative Strength Index)
    rsi = RSIIndicator(close=data['Close'], window=14)
    data['RSI'] = rsi.rsi()

    # MACD (Moving Average Convergence Divergence)
    macd = MACD(close=data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Diff'] = macd.macd_diff()
    
    # Drop NaN values
    data.dropna(inplace=True)
    return data

def prepare_data(data):
    """Prepare features and target variable."""
    features = ['Price_Change', 'MA7', 'MA21', 'Volume_Change', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff']
    X = data[features]
    y = data['Price_Up']
    return X, y

def create_sequences(X, y, time_steps=30):
    """Create sequences of data for LSTM model."""
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

def build_lstm_model(input_shape):
    """Build an LSTM model for binary classification."""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def save_prediction(data, prediction):
    """Save the prediction result to a JSON file."""
    prediction_data = {
        'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'prediction': "UP" if prediction == 1 else "DOWN",
        'openPrice': data['Open'].iloc[-1],
        'closePrice': data['Close'].iloc[-1]
    }
    
    json_file = '../../data/btc_price_predictions.json'
    if os.path.exists(json_file):
        with open(json_file, 'r') as file:
            historical_data = json.load(file)
    else:
        historical_data = []
    
    historical_data.append(prediction_data)
    with open(json_file, 'w') as file:
        json.dump(historical_data, file, indent=4)
    print("Prediction saved to JSON file.")

def main():
    # Step 1: Fetch Bitcoin data
    btc_data = fetch_bitcoin_data()

    # Step 2: Add technical indicators
    btc_data = add_technical_indicators(btc_data)

    # Step 3: Prepare features and target
    X, y = prepare_data(btc_data)

    # Step 4: Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 5: Create sequences for LSTM
    time_steps = 30
    X_seq, y_seq = create_sequences(X_scaled, y, time_steps)

    # Step 6: Split the data into training and testing sets
    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # Step 7: Build and train the LSTM model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    # Step 8: Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Step 9: Make a prediction for tomorrow
    latest_data = X_scaled[-time_steps:].reshape(1, time_steps, -1)
    prediction = model.predict(latest_data)
    prediction_result = 1 if prediction > 0.5 else 0
    print(f"\nThe model predicts that Bitcoin's price will go {'UP' if prediction_result == 1 else 'DOWN'} tomorrow.")

    # Step 10: Save the prediction to a JSON file
    save_prediction(btc_data, prediction_result)

if __name__ == "__main__":
    main()

