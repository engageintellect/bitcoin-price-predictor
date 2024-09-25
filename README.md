# Bitcoin Price Prediction and Visualization

## Description:
This Python project predicts whether the price of Bitcoin will increase or decrease on the next day, using historical price data and machine learning. Additionally, the project visualizes Bitcoin's price movements using candlestick charts along with moving averages for different timeframes.

The project uses:
- **yfinance** to fetch historical Bitcoin price data (Open, Close, High, Low, Volume).
- **RandomForestClassifier** (from scikit-learn) to predict the next day's price movement (up or down).
- **mplfinance** to plot candlestick charts with moving averages.
- **matplotlib** to display all charts simultaneously.

## Features:
1. **Data Fetching**: Downloads Bitcoin historical data from Yahoo Finance starting from 2014.
2. **Feature Engineering**: Calculates daily percentage changes, and computes moving averages (7-day, 21-day, and 200-day) on closing prices.
3. **Machine Learning**: 
   - Trains a Random Forest model to predict whether the Bitcoin price will increase or decrease tomorrow based on the previous day’s features.
   - Uses daily percentage change and moving averages as inputs to the model.
   - Outputs a prediction of whether Bitcoin's price will go UP or DOWN for the next day.
4. **Visualization**: 
   - Generates candlestick charts for Bitcoin price over different timeframes (365 days, 180 days, 30 days).
   - Displays the 200-day moving average where applicable.
   - Shows volume bars along with price movements.

## Requirements:
The following Python libraries are required to run the script:
- `yfinance`: For downloading Bitcoin price data from Yahoo Finance.
- `pandas`: For data manipulation and feature engineering.
- `numpy`: For numerical operations.
- `scikit-learn`: For training the Random Forest model.
- `mplfinance`: For rendering candlestick charts.
- `matplotlib`: For managing chart displays.
- `fastapi`: For creating an API endpoint to serve JSON prediction data.
- `uvicorn`: For serving the FastAPI endpoint.

You can install the required libraries using the following command:
```
pip install yfinance pandas numpy scikit-learn mplfinance matplotlib
```

## Files:
- **main.py**: This is the main Python script that performs the following:
  - Downloads historical Bitcoin data using `yfinance`.
  - Prepares the data by calculating percentage changes and moving averages.
  - Trains a Random Forest classifier on the processed data.
  - Predicts whether Bitcoin's price will go up or down for the next day.
  - Generates candlestick charts for Bitcoin prices over the last 365, 180, and 30 days.

## How to Run:
1. Ensure you have Python installed (version 3.8+ recommended).
2. Install the required libraries (see the **Requirements** section).
3. Run the script by executing the following command:
```
python3 main.py
```
4. The script will:
   - Download and preprocess the historical Bitcoin price data.
   - Train a Random Forest model and predict the next day’s price movement (up or down).
   - Display three candlestick charts:
     - One for the last 365 days with the 200-day moving average.
     - One for the last 180 days.
     - One for the last 30 days.
   
   The prediction (whether Bitcoin’s price will go up or down) will also be printed in the terminal.

## Sample Output:
```
[*********************100%***********************]  1 of 1 completed
The model predicts that Bitcoin's price will go UP tomorrow.
```

Three charts will also be displayed simultaneously showing the Bitcoin price movements over the selected timeframes.

## Notes:
- The 200-day moving average will only appear on charts where at least 200 days of data is available (e.g., the 365-day chart).
- For shorter timeframes (e.g., 180 days and 30 days), the 200-day moving average is omitted.

# bitcoin-price-predictor
