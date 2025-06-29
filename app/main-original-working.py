import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mplfinance as mpf
import matplotlib.pyplot as plt
import json
import os
import time

# === Setup ===
data_file = os.path.expanduser('~/bitcoin-price-predictor/data/yf-data.json')
prediction_file = os.path.expanduser('~/bitcoin-price-predictor/data/btc_price_predictions.json')
today_str = pd.Timestamp.now().strftime('%Y-%m-%d')

# === Fetch BTC data with retry + backoff ===
def fetch_historical_data(start_date: str, end_date: str):
    for attempt in range(3):
        try:
            print(f"[Fetch Attempt {attempt + 1}] Downloading BTC-USD data from {start_date} to {end_date}...")
            df = yf.download('BTC-USD', start=start_date, end=end_date, auto_adjust=False)
            if not df.empty:
                print(f"Successfully fetched {len(df)} rows.")
                return df
            else:
                print(f"[Attempt {attempt + 1}] Empty DataFrame received.")
        except Exception as e:
            print(f"[Attempt {attempt + 1}] Fetch error: {e}")
        delay = 60 * (attempt + 1)
        print(f"Waiting {delay} seconds before retrying...")
        time.sleep(delay)
    print(f"Failed to fetch data for {start_date} to {end_date} after 3 attempts.")
    return None

# === Chunk fetch by year ===
def fetch_in_chunks(start_date, end_date):
    print("Starting chunked fetch (year-by-year)...")
    full_df = pd.DataFrame()
    current_start = pd.to_datetime(start_date)
    final_end = pd.to_datetime(end_date)

    while current_start < final_end:
        current_end = min(current_start + pd.DateOffset(years=1) - pd.Timedelta(days=1), final_end)
        chunk_df = fetch_historical_data(
            current_start.strftime('%Y-%m-%d'),
            (current_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        )
        if chunk_df is not None and not chunk_df.empty:
            full_df = pd.concat([full_df, chunk_df])
        else:
            print(f"Skipping chunk {current_start.strftime('%Y')} due to repeated failures.")
        current_start = current_end + pd.Timedelta(days=1)

    return full_df.drop_duplicates().sort_index()

# === Load or fetch ===
def load_or_fetch_data():
    if os.path.exists(data_file) and os.path.getsize(data_file) > 0:
        try:
            cached_df = pd.read_json(data_file, convert_dates=True)
            cached_df.set_index("Date", inplace=True)
            last_cached_date = cached_df.index[-1].strftime('%Y-%m-%d')
            next_day = (pd.to_datetime(last_cached_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

            if next_day > today_str:
                print("Cache is already up to date. No new data needed.")
                return cached_df

            new_df = fetch_historical_data(next_day, today_str)
            if new_df is None or new_df.empty:
                print("Direct fetch failed. Falling back to chunked fetch for new data.")
                new_df = fetch_in_chunks(next_day, today_str)

            if new_df is not None and not new_df.empty:
                merged_df = pd.concat([cached_df, new_df]).drop_duplicates().sort_index()
                return merged_df
            else:
                print("No new data fetched. Using cached data as is.")
                return cached_df
        except Exception as e:
            print(f"Cache error: {e}")
            print("Falling back to chunked fetch for full history.")
            return fetch_in_chunks('2014-01-01', today_str)
    else:
        return fetch_in_chunks('2014-01-01', today_str)

# === Save data ===
def save_data(df, path):
    try:
        df.reset_index().to_json(path, orient='records', indent=2, date_format='iso')
        print(f"Data saved to {path}")
    except Exception as e:
        print(f"Error saving data: {e}")

# === Save prediction ===
def save_prediction(prediction_data):
    try:
        with open(prediction_file, 'w') as f:
            json.dump(prediction_data, f, indent=4)
        print(f"Prediction saved to {prediction_file}")
    except Exception as e:
        print(f"Error saving prediction: {e}")

# === MAIN ===
merged_df = load_or_fetch_data()
if merged_df is None or merged_df.empty:
    print("No data to process. Exiting.")
    exit()

save_data(merged_df, data_file)

# === Preprocess ===
try:
    merged_df = merged_df.astype({'Open': float, 'High': float, 'Low': float, 'Close': float, 'Volume': float})
    merged_df.dropna(inplace=True)

    merged_df['Price_Change'] = merged_df['Close'].pct_change()
    merged_df['Price_Up'] = np.where(merged_df['Price_Change'] > 0, 1, 0)
    merged_df['MA7'] = merged_df['Close'].rolling(window=7).mean()
    merged_df['MA21'] = merged_df['Close'].rolling(window=21).mean()
    merged_df['MA200'] = merged_df['Close'].rolling(window=200).mean()
    merged_df.dropna(inplace=True)
except Exception as e:
    print(f"Preprocessing error: {e}")
    exit()

# === Model ===
try:
    X = merged_df[['Price_Change', 'MA7', 'MA21']]
    y = merged_df['Price_Up']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    accuracy = rf_model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")
except Exception as e:
    print(f"Model training error: {e}")
    exit()

# === Predict ===
try:
    latest_data = X.iloc[-1].to_frame().T
    prediction = rf_model.predict(latest_data)
    prediction_result = "UP" if prediction == 1 else "DOWN"
    print(f"The model predicts Bitcoin will go {prediction_result} tomorrow.")
except Exception as e:
    print(f"Prediction error: {e}")
    exit()

# === Save prediction ===
if os.path.exists(prediction_file):
    try:
        with open(prediction_file, 'r') as f:
            pred_data = json.load(f)
            if not isinstance(pred_data, list):
                pred_data = []
    except Exception:
        pred_data = []
else:
    pred_data = []

pred_data.append({
    'date': today_str,
    'prediction': prediction_result,
    'openPrice': float(merged_df['Open'].iloc[-1].item()),
    'closePrice': float(merged_df['Close'].iloc[-1].item())
})

save_prediction(pred_data)

# === Plot ===
try:
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    clean_df = merged_df[required_columns].dropna()
    clean_df = clean_df.apply(pd.to_numeric, errors='coerce')
    clean_df.dropna(inplace=True)
    clean_df = clean_df.astype(float)

    # Ensure DatetimeIndex
    if not isinstance(clean_df.index, pd.DatetimeIndex):
        clean_df.index = pd.to_datetime(clean_df.index)

    # Debug info
    print("Sample of clean_df used for plotting:")
    print(clean_df.tail(5))
    print("Dtypes of clean_df:")
    print(clean_df.dtypes)

    valid = clean_df.apply(lambda col: col.map(lambda x: isinstance(x, (int, float)))).all().all()

    if valid:
        mpf.plot(
            clean_df[-365:], type='candle', style='charles', mav=(200),
            title='Bitcoin Price with 200-Day MA (365 Days)',
            volume=True, ylabel='Price (USD)', ylabel_lower='Volume', block=False
        )
        mpf.plot(
            clean_df[-180:], type='candle', style='charles',
            title='Bitcoin Price (180 Days)',
            volume=True, ylabel='Price (USD)', ylabel_lower='Volume', block=False
        )
        mpf.plot(
            clean_df[-30:], type='candle', style='charles',
            title='Bitcoin Price (30 Days)',
            volume=True, ylabel='Price (USD)', ylabel_lower='Volume', block=False
        )
        plt.show()
    else:
        print("Skipping plots due to invalid data types in price columns.")
        bad_rows = clean_df[~clean_df.apply(lambda col: col.map(lambda x: isinstance(x, (int, float)))).all(axis=1)]
        print("Bad rows preventing plot:\n", bad_rows)
except Exception as e:
    print(f"Plotting error: {e}")

