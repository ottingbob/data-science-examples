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
encoder = OneHotEncoder()
targets = encoder.fit_transform(labels).toarray()
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
model.add(Dense(10, input_dim=4, activation="sigmoid"))
# Here is where we have the 3 classes in the output layer and we use
# softmax since we have more than 2 output classes
model.add(Dense(3, input_dim=10, activation="softmax"))

# We can defined the loss function MSE or negative log likelihood
# We can use the ADAM optimizer to find the right adjustments for the weights
# Here the learning rate is a hyper parameter which we can adjust
optimizer = Adam(lr=0.001)

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    # When we use a `softmax` activation function in our output layer we need
    # to use binary cross entropy as our metric
    metrics=["accuracy"],
)

# Batch size defines the number of samples processed before the model is updated
# Number of Epochs is the number of complete passes through the training dataset
# The size of a batch __MUST__ be more than or equal to one and less than or equal
#   to the number of samples in the training dataset
model.fit(feature_train, target_train, batch_size=20, epochs=1_000, verbose=2)

# Multiprocessing allows us to boost up the algorithm
results = model.evaluate(feature_test, target_test, use_multiprocessing=True)

# Print the probabilities associated with each sample in the feature test split
print(model.predict_on_batch(feature_test))

print("Training is finished. The loss and accuracy values are:")
print(results)
