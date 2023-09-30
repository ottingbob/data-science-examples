from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


# This is the cost function
def f(x):
    return x * x


# This is the derivative of the cost function
def df(x):
    return 2 * x


# Here is the actual algorithm
# start of the interval
# end of the interval
# n number of iterations
# alpha learning rate
# momentum allows us to have a smoother path (less oscillation) when finding
#   the next point. This takes the previous value into consideration
def gradient_descent(start, end, n, alpha=0.1, momentum=0.0) -> Tuple:
    # We track the results - x and f(x) values as well
    x_values = []
    y_values = []
    # Generate a random starting point (initial location) between [start, end]
    x = np.random.uniform(start, end)
    for i in range(n):
        # This is the gradient descent formula (based on the derivative)
        x = x - alpha * df(x) - momentum * x
        # Store the x and f(x) values
        x_values.append(x)
        y = f(x)
        y_values.append(y)
        print(f"#{i} f({x}) = {y}")

    return (x_values, y_values)


if __name__ == "__main__":
    start = -1
    end = 1
    step = 0.1
    solutions, scores = gradient_descent(
        start=start,
        end=end,
        n=50,
        alpha=0.1,
        momentum=0.3,
    )
    # Visualize the algorithm
    #
    # Sample input range uniformly at 0.1 increments to plot the function
    inputs = np.arange(start, end + step, step)
    # Create a line plot of the input VS result
    #
    # Here we are visualize the cost function itself
    plt.plot(inputs, f(inputs))
    # Use dots to plot these locations and connect them with a line
    plt.plot(solutions, scores, ".-", color="green")
    plt.title("Gradient Descent")
    plt.show()
