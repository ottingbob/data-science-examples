import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

iris_data = load_iris()

features = iris_data.data
target = iris_data.target

# Reshape the target values
print(target.shape)
# (150,)
# We need to reshape so that we get each of the target values in their own
# individual array so we reshape to make the 1D list into a 2D list
labels = target.reshape(-1, 1)
print(labels.shape)
# (150, 1)

# We have 3 classes so the labels will have 3 values when we perform
# one hot encoding.
#
# first class: (1, 0, 0)
# second class: (0, 1, 0)
# third class: (0, 0, 1)
#
# The importance of doing this is the neural network will perform better
# when we standardize the data between [0, 1]
#
# `sparse` allows us to transform the targets into a one-hot encoded vector
#   otherwise it defaults to a matrix
encoder = OneHotEncoder(sparse=False)
targets = encoder.fit_transform(labels)

# If you don't include `sparse` above you can just convert the matrix values
# to an array like so:
# targets = encoder.fit_transform(labels).toarray()

print(targets.shape)
# (150, 3)

# 20% of dataset is for testing
# 80% of dataset is for training
(
    feature_train,
    feature_test,
    target_train,
    target_test,
) = train_test_split(features, targets, test_size=0.2)

model = Sequential()
# Hidden layer will have 10 neurons and takes the 4 dimensions from the
# feature data as an input
# We use the ReLU activation function to avoid the vanishing gradient problem
model.add(Dense(10, input_dim=4, activation="relu"))
# Here are the hidden layers
model.add(Dense(10, input_dim=10, activation="relu"))
model.add(Dense(10, input_dim=10, activation="relu"))
# Here is where we have the 3 classes in the output layer and we use
# softmax since we have more than 2 output classes
model.add(Dense(3, input_dim=10, activation="softmax"))

# We can defined the loss function MSE or negative log likelihood
# We can use the ADAM optimizer to find the right adjustments for the weights
# Here the learning rate is a hyper parameter which we can adjust
optimizer = Adam(learning_rate=0.001)
model.compile(
    # We use negative log likelihood instead of the mean squared error because
    # we are dealing with a classification problem
    loss="categorical_crossentropy",
    optimizer=optimizer,
    # When we use a `softmax` activation function in our output layer we need
    # to use binary cross entropy as our metric
    metrics=["accuracy"],
)

# Will consider 20 features (batch size) and then update edge weights / model
# parameters accordingly. This will make the training procedure faster
model.fit(feature_train, target_train, epochs=1_000, batch_size=20, verbose=2)

results = model.evaluate(feature_test, target_test)
# We only want the accuracy so we get `results[1]`. If we wanted the loss
# we would look for `results[0]`
print(f"Accuracy on the test dataset {results[1]:0.2f}")
