from datetime import datetime
from math import exp, log, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import fsolve


class call_option:
    # Class for European call options in BSM Model

    def __init__(
        self,
        initial_level: float,
        strike_price: float,
        pricing_date: datetime,
        maturity_date: datetime,
        short_rate: float,
        volatility_factor: float,
    ):
        # S0: initial stack / index level
        # K: strike price
        # t: pricing date
        # M: maturity date
        # r: constant risk-free short date
        # sigma: volatility factor in diffusion term
        self.S0 = float(initial_level)
        self.K = strike_price
        self.t = pricing_date
        self.M = maturity_date
        self.r = short_rate
        self.sigma = volatility_factor

    def update_ttm(self):
        # Update time-to-maturity
        if self.t > self.M:
            raise ValueError("Pricing date later than maturity")
        self.T = (self.M - self.t).days / 365

    def d1(self) -> float:
        # Helper function to derive something...
        d1 = (log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (
            self.sigma * sqrt(self.T)
        )
        return d1

    def value(self) -> float:
        # Return option value
        self.update_ttm()
        d2 = (log(self.S0 / self.K) + (self.r - 0.5 * self.sigma**2) * self.T) / (
            self.sigma * sqrt(self.T)
        )
        value = self.S0 * stats.norm.cdf(self.d1(), 0.0, 1.0) - self.K * exp(
            -self.r * self.T
        ) * stats.norm.cdf(d2, 0.0, 1.0)
        return value

    def vega(self):
        # Return Vega of option
        self.update_ttm()
        vega = self.S0 * stats.norm.pdf(self.d1(), 0.0, 1.0) * sqrt(self.T)
        return vega

    def imp_vol(self, C0, sigma_est=0.2):
        # Return implied volatility given option price.
        # FIXME: How does this work...
        option = call_option(self.S0, self.K, self.t, self.M, self.r, sigma_est)
        option.update_ttm()

        def difference(sigma):
            option.sigma = sigma
            return option.value() - C0

        iv = fsolve(difference, sigma_est)[0]
        return iv


def save_or_load_csv(csv_path_str: str) -> pd.DataFrame:
    csv_path = Path(csv_path_str)
    cols = [
        "Date",
        "SX5P",
        "SX5E",
        "SXXP",
        "SXXE",
        "SXXF",
        "SXXA",
        "DK5F",
        "DKXF",
        "DEL",
    ]
    read_kwargs = dict(
        header=None,
        index_col=0,
        parse_dates=True,
        dayfirst=True,
        # Skip first 4 rows
        skiprows=4,
        sep=";",
        # Use these columns
        names=cols,
    )
    if csv_path.exists():
        es = pd.read_csv(csv_path_str, **read_kwargs)
    else:
        es_url = "http://www.stoxx.com/download/historical_values/hbrbcpe.txt"
        es: pd.DataFrame = pd.read_csv(es_url, **read_kwargs)
        es.to_csv(csv_path, sep=";")
    return es


# Black-Scholes-Merton Implied Volatilities of Call Options on EURO STOXX 50
# Option Quotes from September 30 2014

# Pricing Data
# FIXME: This is not used anywhere...
pdate = pd.Timestamp("30-09-2014")

# EURO STOXX 50 index data
es = save_or_load_csv("./derivatives-with-python/part1/es_data.csv")

# delete the helper column
del es["DEL"]
S0 = es["SX5E"]["30-09-2014"]
r = -0.05

# Option data
data = pd.HDFStore("./derivatives-with-python/part1/es50_option_data.h5", "r")["data"]


# BSM Implied volatilities
def calculate_imp_vols(data: pd.DataFrame) -> pd.DataFrame:
    # Calculate all implied volatilities for the European call options given the tolerance
    # level for moneyness(is this a word..?) of the option
    data = data.copy()
    data["Imp_Vol"] = 0.0
    # tolerance for moneyness
    tol = 0.30
    for row in data.index:
        t = data["Date"][row]
        T = data["Maturity"][row]
        ttm = (T - t).days / 365
        forward = np.exp(r * ttm) * S0
        current_strike = data["Strike"][row]
        if (abs(current_strike - forward) / forward) < tol:
            call = call_option(S0, current_strike, t, T, r, 0.2)
            data["Imp_Vol"][row] = call.imp_vol(data["Call"][row])
    return data


# Graphical Output
def plot_imp_vols(data: pd.DataFrame):
    markers = [".", "o", "^", "v", "x", "D", "d", ">", "<"]
    # Plot the implied volatilities
    maturities = sorted(set(data["Maturity"]))
    plt.figure(figsize=(10, 5))
    for i, mat in enumerate(maturities):
        dat = data[(data["Maturity"] == mat) & (data["Imp_Vol"] > 0)]
        plt.plot(
            dat["Strike"].values,
            dat["Imp_Vol"].values,
            f"b{markers[i]}",
            label=str(mat)[:10],
        )
    plt.grid()
    plt.legend()
    plt.xlabel("strike")
    plt.ylabel("implied volatility")
    plt.show()


if __name__ == "__main__":
    data = calculate_imp_vols(data)
    plot_imp_vols(data)
