from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

STOCK_DATA_FILE = "VaR_stock_data.csv"

# Historical data - START and END dates
start_date = datetime(2014, 1, 1)
end_date = datetime(2018, 1, 1)

# Citigroup Stock Ticker
STOCK_TICKER = "C"


def download_data(stock) -> pd.DataFrame:
    # Name of the stock (key) - stock values (2010-2017) as the values
    stock_data = {}

    # Get closing prices on ticker
    ticker = yf.download(stock, start_date, end_date)

    # Adj closing price takes into account factors such as dividends,
    # stock splits, etc. which handles a more accurate view of a stocks
    # value than just closing price
    stock_data[stock] = ticker["Adj Close"]

    return pd.DataFrame(stock_data)


# This is how we calculate the VaR tomorrow (n=1)
def calculate_value_at_risk(
    position,
    c: float,
    mu: float,
    sigma: float,
):
    value_at_risk = position * (mu - sigma * norm.ppf(1 - c))
    return value_at_risk


# This is how we calculate the VaR for any days in the future
def calculate_value_at_risk_n(
    position,
    c: float,
    mu: float,
    sigma: float,
    n: int,
):
    value_at_risk = position * (mu - sigma * np.sqrt(n) * norm.ppf(1 - c))
    return value_at_risk


if __name__ == "__main__":
    stock_data_fp = Path(STOCK_DATA_FILE)

    if not stock_data_fp.exists():
        stock_data = download_data(STOCK_TICKER)
        stock_data.to_csv(path_or_buf=STOCK_DATA_FILE)
    else:
        stock_data = pd.read_csv(
            filepath_or_buffer=STOCK_DATA_FILE,
            index_col="Date",
        )
        stock_data.index = pd.to_datetime(stock_data.index)

    stock_data["returns"] = np.log(stock_data["C"] / stock_data["C"].shift(1))
    stock_data = stock_data[1:]

    # This is the investment: number of stocks or $s invested
    S = 1_000_000
    # Confidence level - at 95%
    c = 0.95

    # We assume that daily returns are normally distributed
    mu = np.mean(stock_data["returns"])
    sigma = np.std(stock_data["returns"])

    value_at_risk = calculate_value_at_risk(S, c, mu, sigma)
    value_at_risk_10_days = calculate_value_at_risk_n(S, c, mu, sigma, 10)
    print(f"Value at risk is: ${value_at_risk:0.2f}")
    print(f"Value at risk over 10 days is: ${value_at_risk_10_days:0.2f}")
    # print(stock_data)
