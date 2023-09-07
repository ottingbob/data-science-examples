import matplotlib.pyplot as plt
import numpy as np


# r0 is the initial value of the interest rate
# T is the timeframe, in this case 1 year
# N is the number of samples
def vasicek_model(r0, kappa, theta, sigma, T=1, N=1000):
    dt = T / float(N)
    t = np.linspace(0, T, N + 1)
    rates = [r0]

    for _ in range(N):
        # Simulate changes in the interest rate
        dr = kappa * (theta - rates[-1]) * dt + sigma * np.sqrt(dt) * np.random.normal(
            0, 1
        )
        rates.append(rates[-1] + dr)

    return t, rates


def plot_model(t, r):
    plt.plot(t, r)
    plt.xlabel("Time (t)")
    plt.ylabel("Interest Rate r(t)")
    plt.title("Vasicek Model")
    plt.show()


if __name__ == "__main__":
    time, data = vasicek_model(
        r0=1.3,
        kappa=0.9,
        theta=1.5,
        sigma=0.01,
    )
    # We will notice that the graph converges to the theta value
    # Kappa will define how quickly the rate of the graph converges to theta
    # Sigma (std_dev) will determine how much the values fluctuate around
    #   getting to the convergence point
    plot_model(time, data)
