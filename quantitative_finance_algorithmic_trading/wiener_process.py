# TODO: meep meep
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr


# Brownian motion is also a common name for the Wiener Process
def wiener_process(dt: float = 0.1, x0: float = 0, n: int = 1000):
    # W(t=0) = 0
    # Initialize W(t) with zeros
    W = np.zeros(n + 1)

    # We create N+1 timesteps: t=0,1,2,3,4...
    t = np.linspace(x0, n, n + 1)

    # We have to use cumulative sum: on every step the additional value
    # is drawn from a normal distribution with mean 0 and variance dt
    # which equates to: N(0, dt)
    # Also: N(0, dt) = sqrt(dt) * N(0, 1) is usually the formula used
    #
    # We start at 1 since we assume W(t=0) = 0
    #
    # In this case for the `np.random.normal` function we are saying that
    # `0` is our mean and `np.sqrt(dt)` is the standard deviation so we
    # expect points to follow the mean over time but have variance related
    # to +- the standard deviation
    W[1 : n + 1] = np.cumsum(np.random.normal(0, np.sqrt(dt), n))

    return t, W


def plot_process(t, W):
    plt.plot(t, W)
    plt.xlabel("Time(t)")
    plt.ylabel("Wiener-process W(t)")
    plt.title("Wiener-process")
    plt.show()


if __name__ == "__main__":
    time, data = wiener_process()
    plot_process(time, data)
