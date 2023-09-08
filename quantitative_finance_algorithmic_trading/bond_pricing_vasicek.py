import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# We will simulate 1000 r(t) interest rate processes
NUM_SIMULATIONS = 1000
# These are the number of points in a single r(t) process
NUM_POINTS = 200


# Simulate interest rates in the coming year with T=1
def monte_carlo_simulation(x, r0, kappa, theta, sigma, T=1):
    dt = T / float(NUM_POINTS)
    result = []

    for _ in range(NUM_SIMULATIONS):
        rates = [r0]
        for _ in range(NUM_POINTS):
            # Typical mean-reversion stochastic process defined by Ornstein-Uhlenbeck
            # differential equation
            dr = (
                kappa * (theta - rates[-1]) * dt
                + sigma * np.sqrt(dt) * np.random.normal()
            )
            rates.append(rates[-1] + dr)
        result.append(rates)

    simulation_data = pd.DataFrame(result)

    # plt.plot(simulation_data.T)
    # plt.show()

    simulation_data = simulation_data.T
    # Calculate the integral of the r(t) based on the simulated paths
    integral_sum = simulation_data.sum() * dt
    # Present value of a future cash flow
    present_integral_sum = np.exp(-integral_sum)
    # Mean because the integral is the average
    bond_price = x * np.mean(present_integral_sum)
    print(f"Bond price based on Monte-Carlo simulation: {bond_price:0.2f}")


if __name__ == "__main__":
    monte_carlo_simulation(
        1000,
        # Initial value of the interest rate (and starting point of the simulation)
        r0=0.1,
        # The rate at which we approach the mean
        kappa=0.3,
        # The mean we are converging to
        theta=0.3,
        # The variation (std_dev) of the fluctuation around the changes
        sigma=0.03,
    )
