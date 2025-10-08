import yfinance as yf
import pandas as pd
import os
import matplotlib.pyplot as plt

def fetch_stock_data(ticker="AAPL", start="2023-01-01", end="2025-01-01"):
    price_data = yf.download(ticker, start=start, end=end)
    price_data.reset_index(inplace=True)
    price_data = price_data[["Date", "Open", "High", "Low", "Close", "Volume"]]

    #os.makedirs("../data/raw", exist_ok=True)
    price_data.to_csv(f"../data/raw/{ticker}_prices.csv", index=False)
    return price_data


def plot_trades(df, actions):
    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["Close"], label="Close Price", alpha=0.6)

    buy_index = [i for i, a in enumerate(actions) if a == 1]
    sell_index = [i for i, a in enumerate(actions) if a == 2]

    plt.scatter(df["Date"].iloc[buy_index], df["Close"].iloc[buy_index], marker="^")
    plt.scatter(df["Date"].iloc[sell_index], df["Close"].iloc[sell_index], marker="v")

    plt.title("Trading Agent Actions")
    plt.xlabel("Date")
    plt.ylabel("Price ($USD)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    fetch_stock_data()