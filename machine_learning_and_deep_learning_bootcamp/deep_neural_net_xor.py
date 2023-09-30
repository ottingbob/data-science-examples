import numpy as np
from keras.layers import Dense
from keras.models import Sequential

# Why XOR? Because it is a non-linearly separable problem
# XOR problem training samples
training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], "float32")
# XOR problem target values
target_data = np.array([[0], [1], [1], [0]], "float32")

# We can define the neural network layers in a sequential manner
model = Sequential()
# First parameter is output dimension
model.add(Dense(16, input_dim=2, activation="relu"))
# It is overkill to use 7 hidden layers but we are demonstrating how deep
# neural networks are trained under the hood
# The role of the activation function is to make the neural network non-linear
#
# We use the activation function to apply non-linearity to the inputs and the
# edge weights which normally form a linear relationship
#
# Rectified linear activation function - ReLU is the most popular function
# because it makes the gradient a zero or a constant, so it can solve the
# vanishing gradient issue
# If we would use `sigmoid` for the hidden layer activation functions then
# the derivative of the sigmoid would be so small that the update operation
# will not work that much, which would require more epochs to get a similar
# level of accuracy (and minimize loss)
model.add(Dense(16, input_dim=16, activation="relu"))
model.add(Dense(16, input_dim=16, activation="relu"))
model.add(Dense(16, input_dim=16, activation="relu"))
model.add(Dense(16, input_dim=16, activation="relu"))
model.add(Dense(16, input_dim=16, activation="relu"))
model.add(Dense(16, input_dim=16, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

# We can define the loss function MSE or negative log likelihood
#
# Optimizer will find the right adjustments for the weights such as
# SGD, AdaGrad, ADAM...
model.compile(
    loss="mean_squared_error",
    optimizer="adam",
    metrics=["binary_accuracy"],
)

# `epoch` is an iteration over the entire dataset
# verbose 0 is silent, where 1 and 2 are showing results
model.fit(training_data, target_data, epochs=500, verbose=2)
print(model.predict(training_data).round())
# [[0.]
#  [1.]
#  [1.]
#  [0.]]
