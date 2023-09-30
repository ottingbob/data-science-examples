import numpy as np


# Cost function
def f(x, y):
    # Output is a scalar value via a 3D function
    # The scalar is the location we are trying to find with the
    # ADAM optimizer -- in this case `5`
    return x * x + y * x + 5


# Derivative of the cost function
def df(x, y):
    return np.asarray([2.0 * x, 2.0 * y])


# Similar to AdaGrad the `bounds` are a 2D list from the starting
# and ending points
def adam(bounds, n, alpha, beta1, beta2, epsilon=1e-8):
    # Generate an initial point (usually random)
    x = np.asarray([0.8, 0.9])
    # Initialize the `m` and `v` parameters to the size of the
    # number of features
    m = [0.0 for _ in range(bounds.shape[0])]
    v = [0.0 for _ in range(bounds.shape[0])]

    for t in range(1, n + 1):
        # Gradient g(t) so the partial derivatives
        g = df(x[0], x[1])
        # Update every feature independently
        for i in range(x.shape[0]):
            m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
            v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] ** 2
            m_corrected = m[i] / (1.0 - beta1**t)
            v_corrected = v[i] / (1.0 - beta2**t)
            x[i] = x[i] - alpha * m_corrected / (np.sqrt(v_corrected) + epsilon)

        print(f"({x}) - function value: {f(x[0], x[1])}")


if __name__ == "__main__":
    start_bounds = [-1.0, 1.0]
    end_bounds = [-1.0, 1.0]
    adam(
        bounds=np.asarray([start_bounds, end_bounds]),
        n=100,
        alpha=0.05,
        beta1=0.9,
        beta2=0.999,
    )
