from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

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


class ValueAtRiskMonteCarlo:
    # S is the value of our initial investment at t=0
    # n is the value at risk # of days out
    def __init__(self, S, mu, sigma, c, n, iterations):
        self.S = S
        self.mu = mu
        self.sigma = sigma
        self.c = c
        self.n = n
        self.iterations = iterations

    def simulation(self):
        # mean 0 and std_dev 1 in range self.iterations
        rand = np.random.normal(0, 1, [1, self.iterations])
        print(rand)

        # Equation for the S(t) stock price at T
        # The random walk of our initial investment
        stock_price = self.S * np.exp(
            self.n * (self.mu - 0.5 * self.sigma**2)
            + self.sigma * np.sqrt(self.n) * rand
        )

        # Sort prices to determine percentile
        stock_price = np.sort(stock_price)
        # Get percentile based on the confidence interval that we choose
        percentile = np.percentile(stock_price, (1 - self.c) * 100)

        return self.S - percentile


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

    # TODO: Compare `pct_change` to these values...
    # stock_data["returns"] = np.log(stock_data["C"] / stock_data["C"].shift(1))
    # stock_data = stock_data[1:]

    # Calculate daily returns with `pct_change()`
    stock_data["returns"] = stock_data["C"].pct_change()

    # We can assume daily returns to be normally distributed
    # so we can calculate mean / std_dev based on daily returns
    mu = np.mean(stock_data["returns"])
    sigma = np.std(stock_data["returns"])

    model = ValueAtRiskMonteCarlo(
        S=1_000_000,
        mu=mu,
        sigma=sigma,
        # 95% confidence interval
        c=0.95,
        # Value at risk for tomorrow (1 day out)
        n=1,
        # Number of simulations in Monte-Carlo
        iterations=100_000,
    )
    value_at_risk = model.simulation()
    print(f"Value at risk is: ${value_at_risk:0.2f}")
