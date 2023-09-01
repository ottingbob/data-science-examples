from typing import Tuple

import numpy as np
from scipy import stats


def _option_parameters(
    S: float, E: float, T: int, rf: float, sigma: float
) -> Tuple[float, float]:
    d1 = (np.log(S / E) + (rf + sigma * sigma / 2.0) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    print("D1 and D2 parameters:", d1, d2)

    return d1, d2


def call_option_price(S: float, E: float, T: int, rf: float, sigma: float):
    # S is stock price
    # E is strike price
    # T is time of contract or expiry
    # rf is risk free rate
    # sigma is std_dev or volatility of stock
    #
    # First calculate the d1 and d2 parameters
    # in this case `t` is equal to 0
    d1, d2 = _option_parameters(S, E, T, rf, sigma)

    # Use standard normal function N(x) to calculate the price of the option
    return S * stats.norm.cdf(d1) - E * np.exp(-rf * T) * stats.norm.cdf(d2)


def put_option_price(S: float, E: float, T: int, rf: float, sigma: float):
    # S is stock price
    # E is strike price
    # T is time of contract or expiry
    # rf is risk free rate
    # sigma is std_dev or volatility of stock
    #
    # First calculate the d1 and d2 parameters
    # in this case `t` is equal to 0
    d1, d2 = _option_parameters(S, E, T, rf, sigma)

    # Use standard normal function N(x) to calculate the price of the option
    return -S * stats.norm.cdf(-d1) + E * np.exp(-rf * T) * stats.norm.cdf(-d2)


if __name__ == "__main__":
    # Underlying stock price at t=0
    S0 = 100
    # Strike price
    E = 100
    # Expiry 1year = 365 days
    T = 1
    # Risk-free rate
    rf = 0.05
    # Volatility of the underlying stock
    sigma = 0.2

    print(
        "Call option price according to Black-Scholes model:",
        call_option_price(S0, E, T, rf, sigma),
    )
    print(
        "Put option price according to Black-Scholes model:",
        put_option_price(S0, E, T, rf, sigma),
    )
