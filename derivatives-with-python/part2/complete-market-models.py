import math

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from scipy.integrate import quad

# BSM European Call & Put Valuation

# Probability density function of standard normal random variable x
def dN(x: float) -> float:
    return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)


# Cumulative density function of standard normal random variable x
def N(d: float) -> float:
    # -20 lower limit of integration
    # d upper limit of integration
    return quad(lambda x: dN(x), -20, d, limit=50)[0]


# BSM d1 function
def d1f(St: float, K: float, t: int, T: int, r: float, sigma: float) -> float:
    d1 = (math.log(St / K) + (r + 0.5 * sigma**2) * (T - t)) / (
        sigma * math.sqrt(T - t)
    )
    return d1


# Calculate BSM European call option value
# St: stock / index level at time t
# K: strike price
# t: valuation date
# T: date of maturity -- if t = 0; T > t
# r: risk-less short rate constant
# sigma: volatility
def BSM_call_value(
    St: float, K: float, t: int, T: int, r: float, sigma: float
) -> float:
    d1 = d1f(St, K, t, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T - t)
    call_value = St * N(d1) - math.exp(-r * (T - t)) * K * N(d2)
    return call_value


# Calculate BSM European put option value
def BSM_put_value(St: float, K: float, t: int, T: int, r: float, sigma: float) -> float:
    put_value = BSM_call_value(St, K, t, T, r, sigma) - St + math.exp(-r * (T - t)) * K
    return put_value


# Plot European option values for different parameters
def plot_values(fn):
    plt.figure(figsize=(10, 8.3))
    points = 100

    # Model params
    St = 100.0
    K = 100.0
    t = 0.0
    T = 1.0
    r = 0.05
    sigma = 0.2

    # C(K) plot
    plt.subplot(221)
    klist = np.linspace(80, 120, points)
    vlist = [fn(St, K, t, T, r, sigma) for K in klist]
    plt.plot(klist, vlist)
    plt.grid()
    plt.xlabel("strike $K$")
    plt.ylabel("present value")

    # FIXME: The maturity plot DOES NOT line up with the examples from the book however
    # the codebase matches what I have here...
    # C(T) plot
    plt.subplot(222)
    tlist = np.linspace(0.0001, 1, points)
    vlist = [fn(St, K, t, T, r, sigma) for T in tlist]
    print(vlist)
    plt.plot(tlist, vlist)
    plt.grid(True)
    plt.xlabel("maturity $T$")

    # C(r) plot
    plt.subplot(223)
    rlist = np.linspace(0, 0.1, points)
    vlist = [fn(St, K, t, T, r, sigma) for r in rlist]
    plt.plot(rlist, vlist)
    plt.grid(True)
    plt.xlabel("short rate $r$")
    plt.ylabel("present value")
    plt.axis("tight")

    # C(sigma) plot
    plt.subplot(224)
    slist = np.linspace(0.01, 0.5, points)
    vlist = [fn(St, K, t, T, r, sigma) for sigma in slist]
    plt.plot(slist, vlist)
    plt.grid(True)
    plt.xlabel("volatility $\sigma$")

    plt.tight_layout()


# BSM European Call Option Greeks

# BSM DELTA of European call option
def BSM_delta(St: float, K: float, t: int, T: int, r: float, sigma: float) -> float:
    d1 = d1f(St, K, t, T, r, sigma)
    delta = N(d1)
    return delta


# BSM GAMMA of European call option
def BSM_gamma(St: float, K: float, t: int, T: int, r: float, sigma: float) -> float:
    d1 = d1f(St, K, t, T, r, sigma)
    gamma = dN(d1) / (St * sigma * math.sqrt(T - t))
    return gamma


# BSM theta of European call option
def BSM_theta(St: float, K: float, t: int, T: int, r: float, sigma: float) -> float:
    d1 = d1f(St, K, t, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T - t)
    theta = -(
        St * dN(d1) * sigma / (2 * math.sqrt(T - t))
        + r * K * math.exp(-r * (T - t)) * N(d2)
    )
    return theta


# BSM rho of European call option
def BSM_rho(St: float, K: float, t: int, T: int, r: float, sigma: float) -> float:
    d1 = d1f(St, K, t, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T - t)
    rho = K * (T - t) * math.exp(-r * (T - t)) * N(d2)
    return rho


# BSM vega of European call option
def BSM_vega(St: float, K: float, t: int, T: int, r: float, sigma: float) -> float:
    d1 = d1f(St, K, t, T, r, sigma)
    vega = St * dN(d1) * math.sqrt(T - t)
    return vega


def plot_greeks(fn, greek: str):
    # Model params
    St = 100.0
    r = 0.05
    sigma = 0.2
    t = 0.0

    # Greek calculations
    tlist = np.linspace(0.01, 1, 25)
    klist = np.linspace(80, 120, 25)
    # V = np.zeros((len(tlist), len(klist)), dtype=np.cfloat)
    V = np.zeros((len(tlist), len(klist)))
    for j in range(len(klist)):
        for i in range(len(tlist)):
            V[i, j] = fn(St, klist[j], t, tlist[i], r, sigma)

    # 3D plotting
    x, y = np.meshgrid(klist, tlist)
    fig = plt.figure(figsize=(9, 5))
    fig.canvas.manager.set_window_title(greek)
    ax = fig.add_subplot(projection="3d")
    # plot = p3.Axes3D(fig)
    ax.plot_wireframe(x, y, V)
    ax.set_title(greek)
    ax.set_xlabel("strike $K$")
    ax.set_ylabel("maturity $T$")
    ax.set_zlabel(f"{greek}(K, T)")


# CRR Binomal model European option valuation
# otype: `call` or `put`
# M: number of time intervals
def CRR_option_value(
    S0: float, K: float, T: int, r: float, sigma: float, otype: str, M: int = 4
) -> float:
    # Time parameters
    # time interval
    dt = T / M
    # discount per interval
    df = math.exp(-r * dt)

    # Binomial parameters
    # up movement
    u = math.exp(sigma * math.sqrt(dt))
    # down movement
    d = 1 / u
    # martingale branch probability
    q = (math.exp(r * dt) - d) / (u - d)

    # array initialization for index levels
    mu = np.arange(M + 1)
    mu = np.resize(mu, (M + 1, M + 1))
    md = np.transpose(mu)
    mu = u ** (mu - md)
    md = d**md
    S = S0 * mu * md

    # Inner values
    if otype == "call":
        V = np.maximum(S - K, 0)
    else:
        V = np.maximum(K - S, 0)

    z = 0
    # backwards iteration
    for t in range(M - 1, -1, -1):
        V[0 : M - z, t] = (
            q * V[0 : M - z, t + 1] + (1 - q) * V[1 : M - z + 1, t + 1]
        ) * df
        z += 1
    return V[0, 0]


# Plot the CRR option values for increasing number of time intervals M against
# the BSM benchmark value
def plot_covergence(mmin, mmax, step_size):
    # model parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    BSM_benchmark = BSM_call_value(S0, K, 0, T, r, sigma)
    m = range(mmin, mmax, step_size)
    CRR_values = [CRR_option_value(S0, K, T, r, sigma, "call", M) for M in m]
    plt.figure(figsize=(9, 5))
    plt.plot(m, CRR_values, label="CRR values")
    plt.axhline(BSM_benchmark, color="r", ls="dashed", lw=1.5, label="BSM benchmark")
    plt.xlabel("# of binomial steps $M$")
    plt.ylabel("European call option value")
    plt.legend(loc=4)
    plt.xlim(0, mmax)


if __name__ == "__main__":
    # Plot options
    # plot_values(BSM_call_value)
    # plot_values(BSM_put_value)

    # Plot greeks
    """
    greek_map = {
        "delta": BSM_delta,
        "gamma": BSM_gamma,
        "theta": BSM_theta,
        "rho": BSM_rho,
        "vega": BSM_vega,
    }
    for greek_name, greek_fn in greek_map.items():
        plot_greeks(greek_fn, greek_name)
    """

    # FIXME: Again the example matches the code in the github repo but the graph does not...
    plot_covergence(1, 1000, 25)
    plt.show()
