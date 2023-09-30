import numpy as np


# Cost function
def f(x, y):
    # 3 dimensional function
    return x * x + y * y


# Derivative of the cost function
def df(x, y):
    return np.asarray([2.0 * x, 2.0 * y])


# Gradient descent algorithm with AdaGrad
# bounds is a 2D list such as: [[-1, 1], [-1, 1]]
# which holds the starting and ending point of the first and second features
def adaptive_gradient(bounds, n, alpha, epsilon=1e-8):
    # Generate an initial staring point
    solution = np.asarray([0.7, 0.8])
    # G values (sum of the squared past gradients)
    # we have 1 value for every feature (x and y in this case)
    # the value is the sum of the past squared gradients in every iteration
    g_sums = [0.0 for _ in range(bounds.shape[0])]

    # Run the gradient descent
    for _ in range(n):
        # Calculate gradient
        gradient = df(solution[0], solution[1])

        # Update the sum of the squared gradients for all of the features
        # (x and y in this case)
        for i in range(gradient.shape[0]):
            g_sums[i] += gradient[i] ** 2.0

        # Build a solution one variable at a time
        new_solution = []

        # We consider all the features - as we use different learning rates
        # for every given feature
        for i in range(solution.shape[0]):
            # Calculate the step size for the respective feature
            adaptive_alpha = alpha / (np.sqrt(g_sums[i]) + epsilon)
            new_solution.append(solution[i] - adaptive_alpha * gradient[i])

        solution = np.asarray(new_solution)
        solution_value = f(solution[0], solution[1])
        print(f"({solution}) - function value: {solution_value}")


if __name__ == "__main__":
    start_bounds = [-1.0, 1.0]
    end_bounds = [-1.0, 1.0]
    adaptive_gradient(
        bounds=np.asarray([start_bounds, end_bounds]),
        n=200,
        alpha=0.1,
    )
