import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mplfinance as mpf
import matplotlib.pyplot as plt
import json
import os

# Step 1: Fetch Bitcoin data using yfinance
btc_data = yf.download('BTC-USD', start='2014-01-01', end=pd.Timestamp.now().strftime('%Y-%m-%d'), auto_adjust=False)

# Ensure data is not empty
if btc_data.empty:
    print("Error: No data retrieved from yfinance. Exiting script.")
    exit()

# Ensure numeric values and remove NaNs
btc_data[['Open', 'High', 'Low', 'Close', 'Volume']] = btc_data[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
btc_data.dropna(inplace=True)

# Step 2: Prepare the data
btc_data['Price_Change'] = btc_data['Close'].pct_change()
btc_data['Price_Up'] = np.where(btc_data['Price_Change'] > 0, 1, 0)

# Moving Averages
btc_data['MA7'] = btc_data['Close'].rolling(window=7).mean()
btc_data['MA21'] = btc_data['Close'].rolling(window=21).mean()
btc_data['MA200'] = btc_data['Close'].rolling(window=200).mean()

# Drop NaN values
btc_data.dropna(inplace=True)

# Step 3: Define features and target
X = btc_data[['Price_Change', 'MA7', 'MA21']]
y = btc_data['Price_Up']

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 5: Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 5.5: Evaluate the model's accuracy
accuracy = rf_model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 6: Predict if the price will go up or down tomorrow
latest_data = X.iloc[-1].to_frame().T
prediction = rf_model.predict(latest_data)

# Prepare prediction result
prediction_result = "UP" if prediction == 1 else "DOWN"
print(f"The model predicts that Bitcoin's price will go {prediction_result} tomorrow.")

# Step 7: Handle JSON file safely
json_file = '../data/btc_price_predictions.json'

# Load existing JSON data if the file exists
if os.path.exists(json_file):
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
            if not isinstance(data, list):  # Ensure it's a list
                print("Warning: JSON file is invalid. Resetting.")
                data = []
    except (json.JSONDecodeError, ValueError):
        print("Warning: JSON file is corrupted. Resetting.")
        data = []
else:
    data = []

# Append the new prediction instead of overwriting
prediction_data = {
    'date': pd.Timestamp.now().strftime('%Y-%m-%d'),
    'prediction': prediction_result,
    'openPrice': float(btc_data['Open'].iloc[-1]),  # Convert to float
    'closePrice': float(btc_data['Close'].iloc[-1])  # Convert to float
}

data.append(prediction_data)  # Append new prediction to the list

# Write the updated list back to the JSON file
try:
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)
except Exception as e:
    print(f"Error writing JSON file: {e}")


# Step 8: Plot candlestick chart (Last 365 days)
mpf.plot(
    btc_data[-365:], type='candle', style='charles', mav=(200),
    title='Bitcoin Price with 200-Day Moving Average (Last 365 Days)',
    volume=True, ylabel='Price (USD)', ylabel_lower='Volume', block=False
)

# Step 9: Plot last 180 days
mpf.plot(
    btc_data[-180:], type='candle', style='charles',
    title='Bitcoin Price (Last 180 Days)', volume=True,
    ylabel='Price (USD)', ylabel_lower='Volume', block=False
)

# Step 10: Plot last 30 days
mpf.plot(
    btc_data[-30:], type='candle', style='charles',
    title='Bitcoin Price (Last 30 Days)', volume=True,
    ylabel='Price (USD)', ylabel_lower='Volume', block=False
)

# Step 11: Show all charts
plt.show()

