from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as optimization
import yfinance as yf

# On average there are 252 trading days in a year
NUM_TRADING_DAYS = 252
# Generate random `w` weights for different portfolios
NUM_GENERATED_PORTFOLIOS = 10_000

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


def generate_portfolios(
    returns: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []

    for _ in range(NUM_GENERATED_PORTFOLIOS):
        # We need to ensure that the sum of the weights is 1 to express
        # 100% of the portfolio is divided among the available assets
        w = np.random.random(len(stocks))
        # Normalize the `w` weight and make sure sum of items is 1
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * NUM_TRADING_DAYS)
        portfolio_risks.append(
            np.sqrt(
                np.dot(
                    w.T,
                    np.dot(returns.cov() * NUM_TRADING_DAYS, w),
                )
            )
        )

    return (
        np.array(portfolio_weights),
        np.array(portfolio_means),
        np.array(portfolio_risks),
    )


def show_portfolios(returns: pd.DataFrame, volatilities: np.ndarray):
    plt.figure(figsize=(10, 6))
    # The color `c` represents different sharpe ratios
    plt.scatter(volatilities, returns, c=returns / volatilities, marker="o")
    plt.xlabel("Expected Volatility")
    plt.ylabel("Expected Return")
    plt.colorbar(label="Sharpe Ratio")
    plt.show()


def portfolio_stats(weights: np.ndarray, returns: pd.DataFrame) -> np.ndarray:
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(
        np.dot(
            weights.T,
            np.dot(returns.cov() * NUM_TRADING_DAYS, weights),
        )
    )
    return np.array(
        [
            portfolio_return,
            portfolio_volatility,
            portfolio_return / portfolio_volatility,
        ]
    )


# scipy optimize module can find the minimum of a given function
# the maximum of a f(x) is equal to the minimum of -f(x)
def min_function_sharpe(weights: np.ndarray, returns: pd.DataFrame) -> np.ndarray:
    return -portfolio_stats(weights, returns)[2]


# The constraints are that the sum of weights == 1
# f(x) = 0 is the function we are trying to optimize:
#   sum w = 1 -> w = sum - 1
def optimize_portfolio(weights: np.ndarray, returns: pd.DataFrame):
    # TODO: This returns an optimize result

    # `eq` means equality == 0
    # `fun` in this instance helps us maintain that the sum of the weights
    #   will be == 1
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    # the weights can be 1 at most: 1 when 100% of money is invested into a
    # single stock
    bounds = tuple((0, 1) for _ in range(len(stocks)))
    return optimization.minimize(
        fun=min_function_sharpe,
        # Initial position for optimization function
        x0=weights[0],
        args=returns,
        method="SLSQP",
        # Weights can be between values 0 and 1
        bounds=bounds,
        # Sum of weights = 1
        constraints=constraints,
    )


def print_optimal_portfolio(optimum, returns):
    print("Optimal portfolio: ", optimum["x"].round(3))
    print(
        "Expected return, volatility and Sharpe ratio: ",
        portfolio_stats(optimum["x"].round(3), returns),
    )


def show_optimal_portfolio(opt, rets, portfolio_rets, portfolio_vols):
    plt.figure(figsize=(10, 6))
    # The color `c` represents different sharpe ratios
    plt.scatter(
        portfolio_vols, portfolio_rets, c=portfolio_rets / portfolio_vols, marker="o"
    )
    plt.grid(True)
    plt.xlabel("Expected Volatility")
    plt.ylabel("Expected Return")
    plt.colorbar(label="Sharpe Ratio")
    stats = portfolio_stats(opt["x"], rets)
    plt.plot(stats[1], stats[0], "g*", markersize=20.0)
    plt.show()


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
    # show_statistics(log_daily_returns)
    # show_mean_variance(log_daily_returns, np.ndarray([0.2, 0.1, 0.2, 0.2, 0.2, 0.1]))

    weights, means, risks = generate_portfolios(log_daily_returns)
    show_portfolios(means, risks)
    optimum = optimize_portfolio(weights, log_daily_returns)
    print_optimal_portfolio(optimum, log_daily_returns)
    show_optimal_portfolio(optimum, log_daily_returns, means, risks)
