import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NUM_SIMULATIONS = 100


def stock_monte_carlo(S0: float, mu: float, sigma: float, N: int = 1000):
    result = []
    # Number of simulations - possible S(t) realizations of the process
    for _ in range(NUM_SIMULATIONS):
        prices = [S0]
        for _ in range(N):
            # We simulate the change day by day (t=1)
            stock_price = prices[-1] * np.exp(
                (mu - 0.5 * sigma**2) + sigma * np.random.normal()
            )
            prices.append(stock_price)
        result.append(prices)
    simulation_data = pd.DataFrame(result)
    # The given columns will contain the time series for a given simulation
    simulation_data = simulation_data.T

    # Get the mean value of each row (on a day-by-day basis)
    simulation_data["mean"] = simulation_data.mean(axis=1)

    print(simulation_data)
    plt.plot(simulation_data)
    plt.show()

    future_price = round(simulation_data["mean"].tail(1), 2)
    print(f"Prediction for future stock price: {future_price}")


if __name__ == "__main__":
    stock_monte_carlo(S0=50, mu=0.0002, sigma=0.01)
