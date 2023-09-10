import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

HOUSE_PRICES_CSV = "mldl_bootcamp_resources/Datasets/Datasets/house_prices.csv"

house_data = pd.read_csv(HOUSE_PRICES_CSV)

size = house_data["sqft_living"]
price = house_data["price"]

# Machine learning handles arrays not data-frames so we need to convert
# With `np.array(...).reshape()` we use the `-1` arg to indicate we have an
#   unknown dimension and numpy will calculate the number of dimensions for
#   us based on the other param provided
# In this case we are trying to strip out the IDs relating to the index in
#   the pandas dataframe
x = np.array(size).reshape(-1, 1)
y = np.array(price).reshape(-1, 1)

# We use Linear Regression + `fit()` for the training which uses gradient
# descent as the cost function
model = LinearRegression()
model.fit(x, y)

# MSE and R^2 value
regression_model_mse = mean_squared_error(x, y)
print("MSE:", math.sqrt(regression_model_mse))
# R^2 measures how strong of a linear relationship is between the two variables
# The higher number the better with >1 meaning a strong linear correlation
# between the dependent / independent variables
print("R squared value:", model.score(x, y))

# We can get the b values (linear equation values) after the model fit
# y = b0 * x + b1

# This is `b1`
print(model.coef_[0])
# This is `b0`
print(model.intercept_[0])

# Visualize the dataset with the fitted model
plt.scatter(x, y, color="green")
plt.plot(x, model.predict(x), color="black")
plt.title("Linear Regression")
plt.xlabel("Size")
plt.ylabel("Price")
plt.show()

# Predict prices with the model
print("Prediction by the model:", model.predict([[2000]]))


if __name__ == "__main__":
    print("gotem")
