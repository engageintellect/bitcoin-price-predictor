import pandas as pd
import numpy as np
import json
import os
import time
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mplfinance as mpf
import matplotlib.pyplot as plt

# === Setup ===
data_file = os.path.expanduser('~/bitcoin-price-predictor/data/yf-data.json')
prediction_file = os.path.expanduser('~/bitcoin-price-predictor/data/btc_price_predictions.json')
today_str = pd.Timestamp.now().strftime('%Y-%m-%d')


# === Fetch BTC data from CoinGecko ===
def fetch_historical_data(start_date: str, end_date: str):
    print(f"Fetching BTC data from CoinGecko using days='max'...")
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
    try:
        res = requests.get(url, params={
            'vs_currency': 'usd',
            'days': 'max',
            'interval': 'daily'
        }, timeout=10)
        data = res.json()

        if 'prices' not in data:
            raise ValueError("Unexpected CoinGecko response format")

        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('Date', inplace=True)
        df['Open'] = df['price']
        df['High'] = df['price']
        df['Low'] = df['price']
        df['Close'] = df['price']
        df['Volume'] = 0  # Placeholder
        df.drop(columns=['timestamp', 'price'], inplace=True)

        # Truncate to desired end date
        df = df[df.index <= pd.to_datetime(end_date)]
        return df

    except Exception as e:
        print(f"CoinGecko fetch failed: {e}")
        exit(1)



# === Load or initialize cached data ===
use_cache = os.path.exists(data_file) and os.path.getsize(data_file) > 0

if use_cache:
    try:
        cached_df = pd.read_json(data_file, convert_dates=True).set_index("Date")
        last_cached_date = cached_df.index[-1].strftime('%Y-%m-%d')
        new_df = fetch_historical_data(start_date=last_cached_date, end_date=today_str)
        merged_df = pd.concat([cached_df, new_df]).drop_duplicates().sort_index()
    except Exception as e:
        print(f"Failed to read cached data, falling back to full fetch: {e}")
        merged_df = fetch_historical_data(start_date='2014-01-01', end_date=today_str)
else:
    merged_df = fetch_historical_data(start_date='2014-01-01', end_date=today_str)

# Save updated cache
merged_df.to_json(data_file, indent=2, date_format='iso', date_unit='s')


# === Preprocess ===
merged_df = merged_df.astype({'Open': float, 'High': float, 'Low': float, 'Close': float, 'Volume': float})
merged_df.dropna(inplace=True)

merged_df['Price_Change'] = merged_df['Close'].pct_change()
merged_df['Price_Up'] = np.where(merged_df['Price_Change'] > 0, 1, 0)
merged_df['MA7'] = merged_df['Close'].rolling(window=7).mean()
merged_df['MA21'] = merged_df['Close'].rolling(window=21).mean()
merged_df['MA200'] = merged_df['Close'].rolling(window=200).mean()
merged_df.dropna(inplace=True)

# === Train model ===
X = merged_df[['Price_Change', 'MA7', 'MA21']]
y = merged_df['Price_Up']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
accuracy = rf_model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# === Predict tomorrow ===
latest_data = X.iloc[-1].to_frame().T
prediction = rf_model.predict(latest_data)
prediction_result = "UP" if prediction == 1 else "DOWN"
print(f"The model predicts that Bitcoin's price will go {prediction_result} tomorrow.")

# === Append prediction to JSON ===
if os.path.exists(prediction_file):
    try:
        with open(prediction_file, 'r') as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = []
    except Exception:
        data = []
else:
    data = []

data.append({
    'date': today_str,
    'prediction': prediction_result,
    'openPrice': float(merged_df['Open'].iloc[-1]),
    'closePrice': float(merged_df['Close'].iloc[-1])
})

with open(prediction_file, 'w') as f:
    json.dump(data, f, indent=4)


# === Plot charts ===
mpf.plot(
    merged_df[-365:], type='candle', style='charles', mav=(200),
    title='Bitcoin Price with 200-Day Moving Average (Last 365 Days)',
    volume=True, ylabel='Price (USD)', ylabel_lower='Volume', block=False
)

mpf.plot(
    merged_df[-180:], type='candle', style='charles',
    title='Bitcoin Price (Last 180 Days)', volume=True,
    ylabel='Price (USD)', ylabel_lower='Volume', block=False
)

mpf.plot(
    merged_df[-30:], type='candle', style='charles',
    title='Bitcoin Price (Last 30 Days)', volume=True,
    ylabel='Price (USD)', ylabel_lower='Volume', block=False
)

plt.show()

