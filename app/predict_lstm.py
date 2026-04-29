import yfinance as yf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

INDIAN_TICKERS = [
    "TATAMOTORS", "CEATLTD", "RELIANCE", "TCS", "INFY",
    "HDFCBANK", "ICICIBANK", "WIPRO", "SBIN", "ADANIENT",
    "BAJFINANCE", "HINDUNILVR", "MARUTI", "SUNPHARMA", "TITAN",
    "ONGC", "NTPC", "POWERGRID", "COALINDIA", "BPCL",
    "HEROMOTOCO", "DIVISLAB", "DRREDDY", "CIPLA", "APOLLOHOSP",
    "ULTRACEMCO", "GRASIM", "SHREECEM", "ASIANPAINT", "NESTLEIND",
    "BRITANNIA", "DABUR", "MARICO", "GODREJCP", "COLPAL",
    "TATACONSUM", "ITC", "HINDPETRO", "IOC", "GAIL",
    "TECHM", "HCLTECH", "LTIM", "MPHASIS", "PERSISTENT",
    "AXISBANK", "KOTAKBANK", "INDUSINDBK", "BANDHANBNK", "FEDERALBNK",
    "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "NMDC",
    "LT", "SIEMENS", "ABB", "BHEL", "HAL",
    "DMART", "TRENT", "NYKAA", "ZOMATO", "PAYTM",
    "BAJAJFINSV", "SBILIFE", "HDFCLIFE", "STARHEALTH",
    "IRCTC", "INDIGO", "SPICEJET", "GMRINFRA", "ADANIPORTS",
    "TATAPOWER", "ADANIGREEN", "TORNTPOWER", "CESC",
    "PIIND", "UPL", "CHAMBLFERT", "COROMANDEL", "GNFC",
    "MRF", "APOLLOTYRE", "BALKRISIND", "TVSSRICHAK"
]

SHORT_NAME_MAP = {
    "TATA": "TATAMOTORS",
    "CEAT": "CEATLTD",
    "HDFC": "HDFCBANK",
    "SBI": "SBIN",
    "ICICI": "ICICIBANK",
    "KOTAK": "KOTAKBANK",
    "BAJAJ": "BAJFINANCE",
    "INFOSYS": "INFY",
    "WIPRO": "WIPRO",
    "TCS": "TCS",
    "AIRTEL": "BHARTIARTL",
    "ADANI": "ADANIENT",
}

def get_lstm_predictions(ticker="NFLX", days_to_predict=30):

    ticker = ticker.upper().strip()

    # Resolve short names first
    if ticker in SHORT_NAME_MAP:
        ticker = SHORT_NAME_MAP[ticker]

    # Auto add .NS for Indian stocks
    is_indian = ticker in INDIAN_TICKERS
    yf_ticker = ticker + ".NS" if is_indian else ticker

    # Check if local CSV exists
    csv_path = os.path.join(os.path.dirname(__file__), "data", f"{yf_ticker}.csv")

    if os.path.exists(csv_path):
        data = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
        if "Adj Close" in data.columns:
            df = pd.DataFrame(data["Adj Close"])
            df.rename(columns={"Adj Close": "Close"}, inplace=True)
        elif "Close" in data.columns:
            df = pd.DataFrame(data["Close"])
        else:
            raise ValueError("No Close or Adj Close column found in CSV for ticker: " + ticker)
    else:
        # Try downloading with extended date range
        data = yf.download(yf_ticker, start="2010-01-01", end="2025-12-31", progress=False)

        # If empty try without .NS (maybe user typed full yfinance ticker)
        if data.empty and is_indian:
            data = yf.download(ticker, start="2010-01-01", end="2025-12-31", progress=False)

        if data.empty:
            raise ValueError("No data found for ticker: " + ticker + ". Please check the ticker symbol.")

        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if "Adj Close" in data.columns:
            df = pd.DataFrame(data["Adj Close"])
            df.rename(columns={"Adj Close": "Close"}, inplace=True)
        elif "Close" in data.columns:
            df = pd.DataFrame(data["Close"])
        else:
            raise ValueError("No Close or Adj Close column found for ticker: " + ticker)

    if len(df) < 60:
        raise ValueError("Not enough data to train model for ticker: " + ticker)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    X_train, y_train = [], []
    time_step = 50
    for i in range(time_step, len(scaled_data)):
        X_train.append(scaled_data[i-time_step:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    last_seq = X_train[-1:]
    predictions = []
    for _ in range(days_to_predict):
        pred = model.predict(last_seq, verbose=0)[0][0]
        predictions.append(pred)
        last_seq = np.append(last_seq[:, 1:, :], [[[pred]]], axis=1)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    df_pred = pd.DataFrame(predictions, columns=["Predicted"])
    last_date = df.index[-1]
    df_pred.index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict)
    return df, df_pred

if __name__ == "__main__":
    df_hist, df_pred = get_lstm_predictions(ticker="TATAMOTORS", days_to_predict=5)
    print(df_pred)