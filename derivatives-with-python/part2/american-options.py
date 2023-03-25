# Valuation of American Options with CRR Model
# using the Primal Algorithm

import math
from typing import Tuple

import numpy as np


def set_parameters(
    otype: str, M: int
) -> Tuple[float, float, float, int, float, float, float, float, float]:
    # S0: initial stock level
    # T: time to maturity
    # r: short rate
    # sigma: volatility
    if otype == "put":
        S0 = 36.0
        T = 1.0
        r = 0.06
        sigma = 0.2
    elif otype == "call":
        S0 = 36.0
        T = 1.0
        r = 0.05
        sigma = 0.2
    else:
        raise ValueError("Option type not known.")

    # time interval
    dt = T / M
    # discount factor
    df = math.exp(-r * dt)
    # up-movement
    u = math.exp(sigma * math.sqrt(dt))
    # down-movemnet
    d = 1 / u
    # martingale probability
    q = (math.exp(r * dt) - d) / (u - d)

    return S0, T, r, sigma, M, dt, df, u, d, q


# Inner value functions for American put option and short condor spread option
# with American exercise
def inner_value(S, otype: str):
    if otype == "put":
        return np.maximum(40.0 - S, 0)
    elif otype == "call":
        return np.mimimum(40.0, np.maximum(90.0 - S, 0) + np.maximum(S - 110.0, 0))
    else:
        raise ValueError("Option type not known.")


def CRR_option_valuation(otype: str, M: int = 500):
    S0, T, r, sigma, M, dt, df, u, d, q = set_parameters(otype, M)
    # array generation for stock prices
    mu = np.arange(M + 1)
    mu = np.resize(mu, (M + 1, M + 1))
    md = np.transpose(mu)
    mu = u ** (mu - md)
    md = d**md
    S = S0 * mu * md

    # valuation by backwards induction
    # inner value matrix
    h = inner_value(S, otype)
    # value matrix
    V = inner_value(S, otype)
    # continuous values
    C = np.zeros((M + 1), (M + 1))
    # exercise matrix
    ex = np.zeros((M + 1), (M + 1))

    z = 0
    # backwards iteration
    for i in range(M - 1, -1, -1):
        C[0 : M - z, i] = (
            q * V[0 : M - z, i + 1] + (1 - q) * V[1 : M - z + 1, i + 1]
        ) * df
        V[0 : M - z, i] = np.where(
            h[0 : M - z, i] > C[0 : M - z, i], h[0 : M - z, i], C[0 : M - z, i]
        )
        ex[0 : M - z, i] = np.where(h[0 : M - z, i] > C[0 : M - z, i], 1, 0)
        z += 1
    return V[0, 0]


# Valuation of American Options with Least-Squares
# Monte Carlo Primal Algorithm
def american_option_lsm_monte_carlo():
    np.random.seed(150_000)

    S0 = 36.0
    K = 40.0
    T = 1.0
    r = 0.06
    sigma = 0.2

    # simulation parameters
    I = 25_000
    M = 50
    dt = T / M
    df = math.exp(-r * dt)

    # stock price paths
    S = S0 * np.exp(
        np.cumsum(
            (r - 0.5 * sigma**2) * dt
            + sigma * math.sqrt(dt) * np.random.standard_normal((M + 1, I)),
            axis=0,
        )
    )
    S[0] = S0

    # inner values
    h = np.maximum(K - S, 0)

    # initialize present value vector
    V = h[-1]

    # American Option valuation by backwards induction
    for t in range(M - 1, 0, -1):
        rg = np.polyfit(S[t], V * df, 5)
        # continuation values
        C = np.polyval(rg, S[t])
        # exercise decision
        V = np.where(h[t] > C, h[t], V * df)

    # LSM estimator
    V0 = df * np.sum(V) / I
    print(f"American put option value {V0:5.3f}")
