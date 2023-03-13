# Analyze returns from geometric brownian notation
# lol... this example was using pandas 0.17.x....

import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import scipy.stats as scs
import statsmodels.api as sm

mpl.rcParams["font.family"] = "serif"


# Probability density function of a normal random variable x
def dN(x, mu: float, sigma: float) -> float:
    """
    params:
    mu: expected value
    sigma: standard deviation

    returns:
    pdf: value of probability density function
    """
    z = (x - mu) / sigma
    pdf = np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi * sigma**2)
    return pdf


# Simulate a number of years of daily stock quotes
def simulate_gbm():
    # model parameters:
    # initial index level
    S0 = 100.0
    # time horizon
    T = 10.0
    # risk-less short rate
    r = 0.05
    # instantaneous volatility
    vol = 0.2

    # simulation parameters
    np.random.seed(250_000)
    gbm_dates = pd.date_range(start="30-09-2004", end="30-09-2014", freq="B")
    # time steps
    M = len(gbm_dates)
    # index level paths
    I = 1
    # fixed for simplicity
    dt = 1 / 252
    # discount factor
    df = math.exp(-r * dt)

    # stock price paths
    # random numbers
    rand = np.random.standard_normal((M, I))
    # stock matrix
    S = np.zeros_like(rand)
    # initial values
    S[0] = S0
    # stock price paths
    for t in range(1, M):
        S[t] = S[t - 1] * np.exp(
            (r - vol**2 / 2) * dt + vol * rand[t] * math.sqrt(dt)
        )

    # Could probably do this with polars pretty easy using the index as the first column
    gbm = pd.DataFrame(S[:, 0], index=gbm_dates, columns=["index"])
    gbm["returns"] = np.log(gbm["index"] / gbm["index"].shift(1))

    # realized volatility (defined as variance swaps)
    gbm["rea_var"] = 252 * np.cumsum(gbm["returns"] ** 2) / np.arange(len(gbm))
    gbm["rea_vol"] = np.sqrt(gbm["rea_var"])
    gbm = gbm.dropna()
    return gbm


# Return sample stats and normality tests
def print_statistics(data: pd.DataFrame):
    print("Return Sample Statistics")
    print("-" * 50)
    print(f"Mean of Daily Log Returns: {np.mean(data['returns']):9.6f}")
    print(f"Std of Daily Log Returns: {np.std(data['returns']):9.6f}")
    print(f"Mean of Annual Log Returns: {np.mean(data['returns']) * 252:9.6f}")
    print(f"Std of Annual Log Returns: {np.std(data['returns']) * math.sqrt(252):9.6f}")
    print("-" * 50)
    print(f"Skew of Sample Log Returns: {scs.skew(data['returns'])}")
    print(f"Skew Normal Test p-value: {scs.skewtest(data['returns'])[1]}")
    print("-" * 50)
    print(f"Kurt of Sample Log Returns: {scs.kurtosis(data['returns'])}")
    print(f"Kurt Normal Test p-value: {scs.kurtosistest(data['returns'])[1]}")
    print("-" * 50)
    print(f"Normal Test p-value: {scs.normaltest(data['returns'])[1]}")
    print("-" * 50)
    print(f"Realized Volatility: {data['rea_vol'].iloc[-1]}")
    print(f"Realized Variance: {data['rea_var'].iloc[-1]}")


# plot daily quotes and log returns
def quotes_returns(data: pd.DataFrame):
    plt.figure("Daily Quotes and Log returns", figsize=(9, 6))
    # Plot with 3 rows and 1 column and index 1
    plt.subplot(3, 1, 1)
    data["index"].plot()
    plt.ylabel("daily quotes")
    plt.grid(True)
    plt.axis("tight")
    plt.title("Daily Quotes")

    # Plot with 2 rows and 1 column and index 2
    plt.subplot(2, 1, 2)
    data["returns"].plot()
    plt.ylabel("daily log returns")
    plt.grid(True)
    plt.axis("tight")
    plt.title("Daily Log Returns")


# plot histogram of annualized daily log returns
def return_histogram(data: pd.DataFrame):
    plt.figure("Annualized daily log returns Histogram", figsize=(9, 5))
    x = np.linspace(min(data["returns"]), max(data["returns"]), 100)
    plt.hist(np.array(data["returns"]), bins=50, density=True)
    y = dN(x, np.mean(data["returns"]), np.std(data["returns"]))
    plt.plot(x, y, linewidth=2)
    plt.xlabel("log returns")
    plt.ylabel("frequency/probability")
    plt.grid(True)
    plt.title("Annualized daily log returns Histogram")


# plot quantile-quantile chart of annualized daily log returns
def return_qq_chart(data: pd.DataFrame):
    plt.figure("QQ Chart", figsize=(9, 5))
    sm.qqplot(data["returns"], line="s")
    plt.grid(True)
    plt.xlabel("theoretical quantiles")
    plt.ylabel("sample quantiles")
    plt.title("QQ Chart")


# plot realized volatility
def realized_volatility(data: pd.DataFrame):
    plt.figure("Realized Volatility", figsize=(9, 5))
    data["rea_vol"].plot()
    plt.ylabel("realized volatility")
    plt.grid(True)
    plt.title("Realized Volatility")


# plot mean return, volatility and correlation (252 days moving = 1 year)
def rolling_statistics(data: pd.DataFrame):
    plt.figure(figsize=(11, 8))

    plt.subplot(311)
    # mr = data["returns"].rolling(252) * 252
    mr = data["returns"].rolling(window=252).sum()
    mr.plot()
    plt.grid(True)
    plt.ylabel("returns (252d)")
    plt.axhline(mr.mean(), color="r", ls="dashed", lw=1.5)

    plt.subplot(312)
    # vo = pd.rolling_std(data["returns"], 252) * math.sqrt(252)
    vo = data["returns"].rolling(window=252).var()
    vo.plot()
    plt.grid(True)
    plt.ylabel("volatility (252d)")
    plt.axhline(vo.mean(), color="r", ls="dashed", lw=1.5)
    vx = plt.axis()

    plt.subplot(313)
    # co = pd.rolling_corr(mr, vo, 252)
    co = mr.rolling(252).corr(vo)
    co.plot()
    plt.grid(True)
    plt.ylabel("correlation (252d)")
    cx = plt.axis()
    plt.axis([vx[0], vx[1], cx[2], cx[3]])
    plt.axhline(co.mean(), color="r", ls="dashed", lw=1.5)


if __name__ == "__main__":
    gbm = simulate_gbm()
    print_statistics(gbm)
    quotes_returns(gbm)
    return_histogram(gbm)
    return_qq_chart(gbm)
    realized_volatility(gbm)
    rolling_statistics(gbm)
    plt.show()
