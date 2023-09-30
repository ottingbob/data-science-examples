import numpy as np
from keras.layers import Dense
from keras.models import Sequential

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Sequential means we will construct the network on a layer-by-layer basis
# Dense means every neuron in the network is connected to each one in the
# next layer
model = Sequential()
# The hidden layer will contain `4` neurons, and the layer before that
# will feed in input will have `2` neurons or dimensions, and the activation
# function will be the `sigmoid` function
model.add(Dense(4, input_dim=2, activation="sigmoid"))
model.add(Dense(1, input_dim=4, activation="sigmoid"))

# Print out the model edge weights - the weights associated with each
# connection in the neural net
print(model.weights)
# [<tf.Variable 'dense/kernel:0' shape=(2, 4) dtype=float32, numpy=
#  array([[ 0.9279485 ,  0.37704587,  0.15178728,  0.23510408],
#                [-0.31820273, -0.25830698,  0.5155134 , -0.2662952 ]],
#              dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(4, 1) dtype=float32, numpy=
#  array([[-1.0528564 ],
#                [-0.87471926],
#                [-0.80775756],
#                [ 0.4274789 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]

# Loss function measures how close the given neural network is to the ideal
# toward which it is training
# We calculate a value based on the error we observe in the networks
# predictions
model.compile(
    loss="mean_squared_error",
    optimizer="adam",
    metrics=["binary_accuracy"],
)

model.fit(X, y, epochs=30_000, verbose=2)
print("Predictions after the training:")
print(model.predict(X))
# [[6.4926979e-05]
#  [9.9947333e-01]
#  [9.9948663e-01]
#  [7.2035118e-04]]
