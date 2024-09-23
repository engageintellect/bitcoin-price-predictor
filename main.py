import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mplfinance as mpf  # Import mplfinance for candlestick chart
import matplotlib.pyplot as plt  # Import matplotlib to handle show()
import json  # For handling JSON file operations
import os  # For checking if the JSON file exists

# Step 1: Fetch Bitcoin data using yfinance
btc_data = yf.download('BTC-USD', start='2014-01-01', end=pd.Timestamp.now().strftime('%Y-%m-%d'))

# Step 2: Prepare the data
btc_data['Price_Change'] = btc_data['Close'].pct_change()  # Daily percentage change
btc_data['Price_Up'] = np.where(btc_data['Price_Change'] > 0, 1, 0)  # 1 if price went up, 0 if it went down

# Step 2.5: Add the moving averages (7-day, 21-day, and 200-day)
btc_data['MA7'] = btc_data['Close'].rolling(window=7).mean()  # 7-day moving average
btc_data['MA21'] = btc_data['Close'].rolling(window=21).mean()  # 21-day moving average
btc_data['MA200'] = btc_data['Close'].rolling(window=200).mean()  # 200-day moving average

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

# Step 6: Predict if the price will go up or down tomorrow
latest_data = X.iloc[-1].to_frame().T  # Keep the feature names by converting to a DataFrame
prediction = rf_model.predict(latest_data)  # Predict tomorrow's movement

# Prepare the prediction result
prediction_result = "UP" if prediction == 1 else "DOWN"
print(f"The model predicts that Bitcoin's price will go {prediction_result} tomorrow.")

# Step 7: Save the prediction to a JSON file with the current date
prediction_data = {
    'Date': pd.Timestamp.now().strftime('%Y-%m-%d'),
    'Prediction': prediction_result,
    'Close_Price': btc_data['Close'].iloc[-1]  # Add the current close price for reference
}

# Define the JSON file path
json_file = './data/btc_price_predictions.json'

# Check if the JSON file exists
if os.path.exists(json_file):
    # Load the existing data from the JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)
else:
    # If the file does not exist, create an empty list
    data = []

# Append the new prediction to the list
data.append(prediction_data)

# Write the updated list back to the JSON file
with open(json_file, 'w') as file:
    json.dump(data, file, indent=4)

# Step 8: Render candlestick chart for historical data with 200d moving average (Last 365 days)
mpf.plot(
    btc_data[-365:],  # Plot the last 365 days of data
    type='candle',  # Candlestick chart
    style='charles',
    mav=(200),  # Add 200-day moving average
    title='Bitcoin Price with 200-Day Moving Average (Last 365 Days)',
    volume=True,  # Show volume bars
    ylabel='Price (USD)',
    ylabel_lower='Volume',
    block=False  # Ensure the chart does not block execution
)

# Step 9: Render candlestick chart for historical data without 200d MA (Last 180 days)
mpf.plot(
    btc_data[-180:],  # Plot the last 180 days of data
    type='candle',  # Candlestick chart
    style='charles',
    title='Bitcoin Price (Last 180 Days)',  # No 200d MA because it's a shorter timeframe
    volume=True,  # Show volume bars
    ylabel='Price (USD)',
    ylabel_lower='Volume',
    block=False  # Ensure the chart does not block execution
)

# Step 10: Render candlestick chart for last 30 days without 200d MA
mpf.plot(
    btc_data[-30:],  # Plot the last 30 days of data
    type='candle',  # Candlestick chart
    style='charles',
    title='Bitcoin Price (Last 30 Days)',  # No 200d MA because it's a shorter timeframe
    volume=True,  # Show volume bars
    ylabel='Price (USD)',
    ylabel_lower='Volume',
    block=False  # Ensure the chart does not block execution
)

# Step 11: Show all charts at once
plt.show()  # This will display all charts simultaneously

