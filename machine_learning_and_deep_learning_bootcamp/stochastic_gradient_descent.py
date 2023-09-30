from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


# `alpha` defines the learning rate of the optimizer
# `epoch` defines the number of iterations
#
# If batch_size = number of samples (data points) then we are running
#   traditional gradient descent.
# If batch_size = 1 then we are running stochastic gradient descent.
# If 1 < batch_size < num_samples then we are running mini-batch gradient
#   descent.
#
# Therefore we can also refer to this as the mini-batch gradient
# descent algorithm
def sgd(x_values, y_values, alpha=0.01, epoch=20, batch_size=3) -> Tuple:
    # Initial parameters for slope and intercept
    m, b = 0.5, 0.5
    # We want to store the mean squared error terms (MSE)
    error = []

    for _ in range(epoch):
        indexes = np.random.randint(0, len(x_values), batch_size)

        # Take the random x values based on the random generated indices
        xs = np.take(x_values, indexes)
        ys = np.take(y_values, indexes)
        n = len(xs)

        # Now run a standard gradient descent on these values
        f = (b + m * xs) - ys

        # Update the parameters of the linear regression problem:
        #
        # Multiply the x values by the function values and then sum them up
        # and divide by the total number of items
        m += -alpha * 2 * xs.dot(f).sum() / n
        b += -alpha * 2 * f.sum() / n

        # Append the MSE
        error.append(mean_squared_error(y, b + m * x))

    return m, b, error


def plot_regression(x_values, y_values, y_predictions):
    plt.figure(figsize=(8, 6))
    plt.title("Regression with Stochastic Gradient Descent (SGD)")
    plt.scatter(x_values, y_values, label="Data Points")
    # Plot the linear regression model
    plt.plot(x_values, y_predictions, c="#ffa35b", label="Regression")
    plt.legend(fontsize=10)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def plot_mse(mse_values):
    plt.figure(figsize=(8, 6))
    # Plot the MSE values
    plt.plot(range(len(mse_values)), mse_values)
    plt.title("Stochastic Gradient Descent Error")
    plt.legend(fontsize=10)
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.show()


if __name__ == "__main__":
    x = pd.Series(list(range(1, 8)))
    y = pd.Series([1, 1.5, 3.5, 5, 5.2, 7.9, 6])

    slope, intercept, mse = sgd(x, y, alpha=0.01, epoch=100, batch_size=3)
    # The model is the linear regression model
    model_predictions = slope * x + intercept

    # Show the results
    print(f"Slope and intercept: {slope} - {intercept}")
    print(f"MSE: {mean_squared_error(y, model_predictions)}")
    plot_regression(x, y, model_predictions)
    plot_mse(mse)
