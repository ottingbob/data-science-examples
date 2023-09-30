import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

# CIFAR-10 dataset has 60_000 32x32 color images in 10 classes with 6_000
# images per class. There are 50_000 training and 10_000 testing images
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print("X_train original shape", X_train.shape)
# X_train original shape (50000, 32, 32, 3)
print("y_train original shape", y_train.shape)
# y_train original shape (50000, 1)
print("X_test original shape", X_test.shape)
# X_test original shape (10000, 32, 32, 3)
print("y_test original shape", y_test.shape)
# y_test original shape (10000, 1)

# One-hot encoding for the labels (1,2, ...) will be replaced with arrays of
# 1s and 0s
target_train = to_categorical(y_train, 10)
target_test = to_categorical(y_test, 10)
print(target_train.shape)
# (50000, 10)

# Normalize the dataset (test and training set as well)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

# Build the Convolutional Neural Network (CNN)
model = Sequential()


def original_model():
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            # Deals with edge weight initialization
            kernel_initializer="he_uniform",
            # Padding evenly to the left and right of inputs so that the output has
            # the same dimensions as the input
            padding="same",
            input_shape=(32, 32, 3),
        )
    )
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            # Deals with edge weight initialization
            kernel_initializer="he_uniform",
            # Padding evenly to the left and right of inputs so that the output has
            # the same dimensions as the input
            padding="same",
        )
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu",
            # Deals with edge weight initialization
            kernel_initializer="he_uniform",
            # Padding evenly to the left and right of inputs so that the output has
            # the same dimensions as the input
            padding="same",
        )
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation="relu",
            # Deals with edge weight initialization
            kernel_initializer="he_uniform",
            # Padding evenly to the left and right of inputs so that the output has
            # the same dimensions as the input
            padding="same",
        )
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    # Now we add the densely connected neural network as usual
    model.add(Dense(128, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(10, activation="softmax"))


def conv2d_layer(filters: int):
    return Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        activation="relu",
        kernel_initializer="he_uniform",
        padding="same",
    )


def upgraded_model():
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            # Deals with edge weight initialization
            kernel_initializer="he_uniform",
            # Padding evenly to the left and right of inputs so that the output has
            # the same dimensions as the input
            padding="same",
            input_shape=(32, 32, 3),
        )
    )
    model.add(BatchNormalization())
    model.add(conv2d_layer(32))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(conv2d_layer(64))
    model.add(BatchNormalization())
    model.add(conv2d_layer(64))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(conv2d_layer(128))
    model.add(BatchNormalization())
    model.add(conv2d_layer(128))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    # Now we add the densely connected neural network as usual
    model.add(Dense(128, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(10, activation="softmax"))


# original_model()

# We add batch normalization and dropout to apply regularization to our model
# Regularization reduces overfitting on the training data and improves the
# performance of the algorithm on the test set
#
# Weight Decay Technique:
# It aims to prevent edge weights from getting too large
# L1 regularization: sum of the absolute weights
# L2 regularization: sum of the squared weights
# Technically this means we add some additional value to the loss function
#
# Dropout Technique:
# Prevents complex co-adaptations on the training data
# Randomly drops some neurons during the training process
# This is good because the neural network layers co-adapt to correct mistakes
# from the prior layers
# Because of the dropout the weights of the neural network will be larger.
# It is a good approach to re-scale the weight by the chosen dropout rate
#
# This should achieve close to 83% accuracy but takes 1h30m to train so
# I have opted to not run it.
upgraded_model()

# Training the model
optimizer = SGD(learning_rate=0.001, momentum=0.95)
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    # When we use a `softmax` activation function in our output layer we need
    # to use binary cross entropy as our metric
    metrics=["accuracy"],
)

model.fit(
    X_train,
    target_train,
    batch_size=64,
    # Consider every image in the dataset 50 times
    epochs=50,
    validation_data=(X_test, target_test),
    verbose=1,
)

# Evaluate the model
score = model.evaluate(X_test, target_test)
print(f"Test accuracy: {score[1]:0.2f}")
# SO this performed pretty bad so we will use regularization to try and make
# it better
# This is the output from the `original_model` so it can be improved:
"""
Epoch 49/50
782/782 [==============================] - 58s 75ms/step - loss: 1.8730e-04 - accuracy: 1.0000 - val_loss: 2.4350 - val_accuracy: 0.7403
Epoch 50/50
782/782 [==============================] - 63s 80ms/step - loss: 1.7606e-04 - accuracy: 1.0000 - val_loss: 2.4497 - val_accuracy: 0.7399
313/313 [==============================] - 2s 7ms/step - loss: 2.4497 - accuracy: 0.7399
Test accuracy: 0.74
"""
