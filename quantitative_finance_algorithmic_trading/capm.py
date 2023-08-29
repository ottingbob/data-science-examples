from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# Market interest rate
RISK_FREE_RATE = 0.05
# We consider monthly returns and we want to calculate annual return
MONTHS_IN_YEAR = 12

# Stocks we are going to handle
stocks = ["IBM", "^GSPC"]

# Historical data - START and END dates
start_date = "2010-01-01"
end_date = "2017-01-01"


class CAPM:
    stock_data_file = "capm_stock_data.csv"

    def __init__(self, stocks: pd.DataFrame, start_date: str, end_date: str):
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date

    def download_data(self) -> pd.DataFrame:
        # Name of the stock (key) - stock values (2010-2017) as the values
        stock_data = {}

        for stock in self.stocks:
            # Closing prices on ticker
            # ticker = yf.Ticker(stock)
            ticker = yf.download(stock, self.start_date, self.end_date)
            # stock_data[stock] = ticker.history(start=start_date, end=end_date)["Close"]

            # Adj closing price takes into account factors such as dividends,
            # stock splits, etc. which handles a more accurate view of a stocks
            # value than just closing price
            stock_data[stock] = ticker["Adj Close"]

        return pd.DataFrame(stock_data)

    def initialize(self):
        # Save the dataframe locally in case we want to try and run it again
        stock_data_fp = Path(self.stock_data_file)

        if not stock_data_fp.exists():
            stock_data = self.download_data()
            stock_data.to_csv(path_or_buf=self.stock_data_file)
        else:
            stock_data = pd.read_csv(
                filepath_or_buffer=self.stock_data_file, index_col="Date"
            )
            stock_data.index = pd.to_datetime(stock_data.index)

        self.stock_data = stock_data
        # We use monthly returns instead of daily returns. These are better
        # for long-term models in that they are at least approximately
        # normally distributed.
        # Daily returns are better for short-term tactical forecasting
        monthly_stock_data = stock_data.resample("M").last()

        self.data = pd.DataFrame(
            {
                "s_adjclose": monthly_stock_data[self.stocks[0]],
                "m_adjclose": monthly_stock_data[self.stocks[1]],
            }
        )

        # Get logarithmic monthly returns
        self.data[["s_returns", "m_returns"]] = np.log(
            self.data[["s_adjclose", "m_adjclose"]]
            / self.data[["s_adjclose", "m_adjclose"]].shift(1)
        )
        # Remove the first row to remove `NaN` entries due to shift
        self.data = self.data[1:]

    def calculate_beta(self):
        # Create the covariance matrix where the diagonal items are the
        # variances and the other values are the covariances
        # The matrix is symmetric in that: cov[0, 1] = cov[1, 0]
        covariance_matrix = np.cov(self.data["s_returns"], self.data["m_returns"])
        print(covariance_matrix)
        # Calculate beta according to CAPM Beta formula
        # beta = covariance(IBM, S&P500) / variance(S&P500)
        beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
        print("Beta from formula:", beta)
        # Here are the meanings related to the given beta values:
        # B = 1: stock moves exactly with the market
        #   Expect the risk free return
        # B > 1: stock market risk is higher than that of an average stock
        #   Higher return and riskier
        # B < 1: stock market risk is lower than that of an average stock
        #   Less return and safer

    def regression(self):
        # Use linear regression to fit a line to the data
        # [stock_returns, market_returns] - slope is the beta
        beta, alpha = np.polyfit(self.data["m_returns"], self.data["s_returns"], deg=1)
        print("Beta from regression: ", beta)
        print("Alpha from regression: ", alpha)
        # Calculate the expected return according to the CAPM formula.
        # We are after the annual return so we multiply by 12
        expected_return = RISK_FREE_RATE + beta * (
            self.data["m_returns"].mean() * MONTHS_IN_YEAR - RISK_FREE_RATE
        )
        print("Expected return: ", expected_return)
        # In this case we can expect a 9% profit
        # Expected return:  0.09011312371676534
        self.plot_regression(alpha, beta)

    def plot_regression(self, alpha: float, beta: float):
        fig, axis = plt.subplots(1, figsize=(10, 8))
        axis.scatter(
            self.data["m_returns"], self.data["s_returns"], label="Data Points"
        )
        axis.plot(
            self.data["m_returns"],
            beta * self.data["m_returns"] + alpha,
            color="red",
            label="CAPM Line",
        )
        plt.title("Capital Asset Pricing Model, Finding Alphas and Betas")
        plt.xlabel("Market Return $R_m$", fontsize=18)
        plt.ylabel("Stock Return $R_a$")
        plt.text(0.08, 0.05, r"$R_a = \beta * R_m + \alpha$", fontsize=18)
        plt.legend()
        plt.grid(True)
        plt.show()


def plot_distributions_histogram(stock_data: pd.DataFrame):
    stock_data["Price"] = stock_data["IBM"]
    stock_data["Price"] = np.log(stock_data["Price"] / stock_data["Price"].shift(1))
    stock_data = stock_data[1:]

    # Only leave the price column
    stock_data.drop("^GSPC", axis=1, inplace=True)
    stock_data.drop("IBM", axis=1, inplace=True)

    plt.hist(stock_data, bins=700)
    stock_variance = stock_data.var()
    stock_mean = stock_data.mean()
    # Standard deviation or volatility
    sigma = np.sqrt(stock_variance)
    # Plot the normal dist curve with respect to 5 standard deviations on
    # each side
    # The fat tails on this line indicate that extreme events occur much more
    # frequently in reality than what a normal distribution would predict
    x = np.linspace(stock_mean - 5 * sigma, stock_mean + 5 * sigma, 100)
    plt.plot(x, norm.pdf(x, stock_mean, sigma))
    plt.show()


if __name__ == "__main__":
    capm = CAPM(
        stocks=["IBM", "^GSPC"],
        start_date=start_date,
        end_date=end_date,
    )
    capm.initialize()
    capm.calculate_beta()
    # capm.regression()

    plot_distributions_histogram(capm.stock_data)
