import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from ta.momentum import RSIIndicator
from ta.trend import MACD
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

def train_logistic_regression(X_train, y_train):
    """Train a Logistic Regression model."""
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model accuracy and generate a classification report."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return accuracy

def cross_validate_model(model, X, y):
    """Perform cross-validation using TimeSeriesSplit."""
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X, y, cv=tscv)
    print(f"Time Series Cross-Validation Accuracy: {np.mean(cv_scores):.2f}")

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

    # Step 5: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

    # Step 6: Train the Logistic Regression model
    model = train_logistic_regression(X_train, y_train)

    # Step 7: Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Step 8: Cross-validate the model
    cross_validate_model(model, X_scaled, y)

    # Step 9: Make a prediction for tomorrow
    latest_data = X_scaled[-1].reshape(1, -1)
    prediction = model.predict(latest_data)
    prediction_result = "UP" if prediction == 1 else "DOWN"
    print(f"\nThe model predicts that Bitcoin's price will go {prediction_result} tomorrow.")

    # Step 10: Save the prediction to a JSON file
    save_prediction(btc_data, prediction)

if __name__ == "__main__":
    main()

