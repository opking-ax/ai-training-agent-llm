import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(ticker="AAPL", start="2023-01-01", end="2025-01-01"):
    price_data = yf.download(ticker, start=start, end=end)
    price_data.reset_index(inplace=True)
    price_data = price_data[["Date", "Open", "High", "Low", "Close", "Volume"]]

    #os.makedirs("../data/raw", exist_ok=True)
    price_data.to_csv(f"../data/raw/{ticker}_prices.csv", index=False)
    return price_data

if __name__ == "__main__":
    fetch_stock_data()