from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as optimization
import yfinance as yf

# On average there are 252 trading days in a year
NUM_TRADING_DAYS = 252

stock_data_file = "markowitz_stock_data.csv"

# Stocks we are going to handle
stocks = ["AAPL", "WMT", "TSLA", "GE", "AMZN", "DB"]

# Historical data - START and END dates
start_date = "2010-01-01"
end_date = "2017-01-01"


def download_data() -> pd.DataFrame:
    # Name of the stock (key) - stock values (2010-2017) as the values
    stock_data = {}

    for stock in stocks:
        # Closing prices on ticker
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)["Close"]

    return pd.DataFrame(stock_data)


def show_data(data: pd.DataFrame) -> None:
    data.plot(figsize=(10, 5))
    plt.show()


def calculate_return(data: pd.DataFrame):
    # This is able to do something like the following:
    # ln( S(t+1) / S(t) )
    #
    # We use log returns to normalize the data in the return dataset
    # which measures all variables in a comparable metric
    log_return = np.log(data / data.shift(1))

    # Remove the first row since it will be `NaN` since we are shifting
    # the data above
    return log_return[0:]


def show_statistics(returns: pd.DataFrame):
    # Instead of daily metrics we are after annual metrics
    #
    # Mean of annual return
    print(returns.mean() * NUM_TRADING_DAYS)
    # Covariance of annual return
    print(returns.cov() * NUM_TRADING_DAYS)


# `weights` define how much of the portfolio is allocated to a given stock
def show_mean_variance(returns: pd.DataFrame, weights: pd.DataFrame):
    # We are after the annual return
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    # Multiply matrix with a vector
    portfolio_volatility = np.sqrt(
        np.dot(
            weights.T,
            np.dot(returns.cov() * NUM_TRADING_DAYS, weights),
        )
    )
    print("Expected portfolio mean (return):", portfolio_return)
    print("Expected portfolio volatility (standard deviation):", portfolio_volatility)


if __name__ == "__main__":
    # Save the dataframe locally in case we want to try and run it again
    stock_data_fp = Path(stock_data_file)

    if not stock_data_fp.exists():
        stock_data_df = download_data()
        stock_data_df.to_csv(path_or_buf=stock_data_file)
    else:
        stock_data_df = pd.read_csv(
            filepath_or_buffer=stock_data_file, index_col="Date"
        )

    print(stock_data_df)
    # show_data(stock_data_df)
    log_daily_returns = calculate_return(stock_data_df)
    show_statistics(log_daily_returns)
