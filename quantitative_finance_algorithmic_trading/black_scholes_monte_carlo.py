from typing import Any, Tuple

import numpy as np


class OptionPricing:

    # S0 is initial stock price
    # E is strike price
    # T is time of contract or expiry
    # rf is risk free rate
    # sigma is std_dev or volatility of stock
    def __init__(self, S0, E, T, rf, sigma, iterations: int):
        self.S0 = S0
        self.E = E
        self.T = T
        self.rf = rf
        self.sigma = sigma
        self.iterations = iterations

    def _option_base_data(self) -> Tuple[Any]:
        # We have 2 columns: First with 0s and the second column will store
        # the payoff
        # We need the first column of 0s: payoff function is max(0, S-E) for
        # the call option
        option_data = np.zeros([self.iterations, 2])
        # print(option_data)

        # Dimensions: 1 dimensional array with as many items as iterations
        rand = np.random.normal(0, 1, [1, self.iterations])
        # print(rand)

        # Equation for the S(t) stock price at T
        stock_price = self.S0 * np.exp(
            self.T * (self.rf - 0.5 * self.sigma**2)
            + self.sigma * np.sqrt(self.T) * rand
        )
        # print(stock_price)

        return option_data, rand, stock_price

    def call_option_simulation(self):
        option_data, rand, stock_price = self._option_base_data()

        # We need S-E because we have to calculate the max(S-E, 0)
        option_data[:, 1] = stock_price - self.E
        # print(option_data)

        # Average for the Monte-Carlo simulation
        # `max()` returns the max(0, S-E) according to the formula
        # This is the average value we are after
        average = np.sum(np.amax(option_data, axis=1)) / float(self.iterations)

        # Have to use the exp(-r * T) discount factor
        return np.exp(-1.0 * self.rf * self.T) * average

    def put_option_simulation(self):
        option_data, rand, stock_price = self._option_base_data()

        # We need S-E because we have to calculate the max(S-E, 0)
        option_data[:, 1] = self.E - stock_price
        # print(option_data)

        # Average for the Monte-Carlo simulation
        # `max()` returns the max(0, S-E) according to the formula
        # This is the average value we are after
        average = np.sum(np.amax(option_data, axis=1)) / float(self.iterations)

        # Have to use the exp(-r * T) discount factor
        return np.exp(-1.0 * self.rf * self.T) * average


if __name__ == "__main__":
    model = OptionPricing(
        S0=100,
        E=100,
        # 1 year expiration
        T=1,
        rf=0.05,
        sigma=0.2,
        iterations=1000,
        # iterations=5,
    )
    call_option_value = model.call_option_simulation()
    print(f"Value of the call option is: ${call_option_value:.2f}")
    put_option_value = model.put_option_simulation()
    print(f"Value of the put option is: ${put_option_value:.2f}")
