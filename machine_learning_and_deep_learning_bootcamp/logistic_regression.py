import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

# Logistic regression is a linear regression on the logit transformation

x1 = np.array([0, 0.6, 1.1, 1.5, 1.8, 2.5, 3, 3.1, 3.9, 4, 4.9, 5, 5.1])
y1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

x2 = np.array([3, 3.8, 4.4, 5.2, 5.5, 6.5, 6, 6.1, 6.9, 7, 7.9, 8, 8.1])
y2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

X = np.concatenate([x1, x2])
y = np.concatenate([y1, y2])

# We need to reshape X in order for the logistic regression to accept the data
# in the desired format
X = np.reshape(X, (-1, 1))
print(X)
print(y)

plt.plot(x1, y1, "ro", color="blue")
plt.plot(x2, y2, "ro", color="red")

# This will again use the Gradient Descent algorithm to find the `b0` and `b1`
# params under the hood when computing the regression equation
model = LogisticRegression()
model.fit(X, y)

print("b0 is:", model.intercept_)
print("b1 is:", model.coef_)


# Helper method to plot logistic equation via sklearn implementation
def logistic(classifier, x):
    return 1 / (1 + np.exp(-(model.intercept_ + model.coef_ * x)))


for i in range(1, 120):
    plt.plot(i / 10.0 - 2, logistic(model, i / 10.0), "ro", color="green")

# Horizontal axis from [-2, 10]
# Vertical axis from [-0.5, 2]
plt.axis([-2, 10, -0.5, 2])
plt.show()

prediction = model.predict([[1]])
print("Prediction:", prediction)

# This will show the probabilities of which binary outcome this prediction
# falls into:
# [[0.00391681 0.99608319]]
print(model.predict_proba([[10]]))
