import matplotlib.pyplot as plt
from keras.datasets import mnist
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
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

# We can load the MNIST dataset from keras datasets
# It will have 60_000 training samples and 10_000 test samples
# They will be 28x28 pixel images
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("X_train original shape", X_train.shape)
# X_train original shape (60000, 28, 28)
print("y_train original shape", y_train.shape)
# y_train original shape (60000,)
print("X_test original shape", X_test.shape)
# X_test original shape (10000, 28, 28)
print("y_test original shape", y_test.shape)
# y_test original shape (10000,)


def plot_random_sample():
    # Plot a grayscale image with the respective label
    plt.imshow(X_train[0], cmap="gray")
    plt.title("Class " + str(y_train[0]))
    plt.show()


# Tensorflow can handle the format: (batch, height, width, channel)
# channel defines 3 for RGB or 1 for grayscale
feature_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
feature_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

feature_train = feature_train.astype("float32")
feature_test = feature_test.astype("float32")

# Very similar to min-max normalization here we transform the values
# within the range [0,1] as usual
feature_train /= 255
feature_test /= 255

# We have 10 output classes we want to end up with so we use one-hot
# encoding to transform the numerical labels into arrays with 0 or 1
# depending on which digit is in the label
target_train = to_categorical(y_train, 10)
target_test = to_categorical(y_test, 10)

# Build the Convolutional Neural Network (CNN)
model = Sequential()

# Input is a 28x28 pixel image
#
# 32 is the number of filters - (3,3) size of the filter
# Create the feature maps from the filters (kernels)
#
# `input_shape` defines the pixel image and the channel `1`
#
# The tuple (3,3) is automatically moved by a stride of (1,1)
# where it will move to the right and downward every iteration
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
model.add(Activation("relu"))

# Normalize the activations in the previous layer after the convolutional phase
# Transformation maintains the mean activation close to 0 std close to 1
# The scale of each dimension remains the same
# Reduces the running time of training significantly
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))

# Get the maximum values from the feature maps on a 2x2 grid. This selects the
# most relevant features and helps us deal with spacial invariance
# IE: the image is scaled or rotated
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))

model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Move the values into a 1D array to be able to prepare it for a feed-forward
# neural network
# This allows use to preprocess the data and use ANN with just the relevant
# feature values
# We use the multilayer neural network to learn the non-linear combinations
# of these important features
model.add(Flatten())
model.add(BatchNormalization())
# Feed forward related hidden layer
model.add(Dense(512))
model.add(Activation("relu"))
model.add(BatchNormalization())
# Dropout regularization which helps to avoid overfitting
# This is an inexpensive regularization method
# We set the activation of a given neuron to be 0 temporarily
# This works well with stochastic gradient descent
# We apply the dropout in the hidden layer exclusively and omit neurons with
# p(~0.5) probability
# This helps prevent coadaptation among detectors, which helps drive better
# generalization in given models
# This is less effective as the number of training samples rises up
# such as over tens of millions
model.add(Dropout(0.3))

# And now we have our output layer of the CNN
model.add(Dense(10, activation="softmax"))

model.summary()
"""
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #
=================================================================
conv2d (Conv2D)             (None, 26, 26, 32)        320

activation (Activation)     (None, 26, 26, 32)        0

batch_normalization (Batch  (None, 26, 26, 32)        128
Normalization)

conv2d_1 (Conv2D)           (None, 24, 24, 32)        9248

activation_1 (Activation)   (None, 24, 24, 32)        0

max_pooling2d (MaxPooling2  (None, 12, 12, 32)        0
D)

batch_normalization_1 (Bat  (None, 12, 12, 32)        128
chNormalization)

conv2d_2 (Conv2D)           (None, 10, 10, 64)        18496

activation_2 (Activation)   (None, 10, 10, 64)        0

batch_normalization_2 (Bat  (None, 10, 10, 64)        256
chNormalization)

conv2d_3 (Conv2D)           (None, 8, 8, 64)          36928

activation_3 (Activation)   (None, 8, 8, 64)          0

max_pooling2d_1 (MaxPoolin  (None, 4, 4, 64)          0
g2D)

flatten (Flatten)           (None, 1024)              0

batch_normalization_3 (Bat  (None, 1024)              4096
chNormalization)

dense (Dense)               (None, 512)               524800

activation_4 (Activation)   (None, 512)               0

batch_normalization_4 (Bat  (None, 512)               2048
chNormalization)

dropout (Dropout)           (None, 512)               0

dense_1 (Dense)             (None, 10)                5130

=================================================================
Total params: 601578 (2.29 MB)
Trainable params: 598250 (2.28 MB)
Non-trainable params: 3328 (13.00 KB)
_________________________________________________________________
"""

# Multiclass classification: cross-entropy loss-function with ADAM optimizer
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    # When we use a `softmax` activation function in our output layer we need
    # to use binary cross entropy as our metric
    metrics=["accuracy"],
)


def train_and_evaluate():
    model.fit(
        feature_train,
        target_train,
        batch_size=128,
        # Consider every image in the dataset 2 times
        epochs=2,
        validation_data=(feature_test, target_test),
        verbose=1,
    )

    score = model.evaluate(feature_test, target_test)
    print(f"Test accuracy: {score[1]:0.2f}")
    """
    Epoch 1/2
    469/469 [==============================] - 34s 69ms/step - loss: 0.0965 - accuracy: 0.9707 - val_loss: 0.6565 - val_accuracy: 0.7660
    Epoch 2/2
    469/469 [==============================] - 32s 69ms/step - loss: 0.0334 - accuracy: 0.9896 - val_loss: 0.0285 - val_accuracy: 0.9913
    313/313 [==============================] - 2s 5ms/step - loss: 0.0285 - accuracy: 0.9913
    Test accuracy: 0.99
    """


# Apply data augmentation to help reduce overfitting
# We don't apply augmentation on the test images, but we need them in the same
# batched format
train_generator = ImageDataGenerator(
    rotation_range=7,
    width_shift_range=0.05,
    shear_range=0.05,
    height_shift_range=0.07,
    zoom_range=0.05,
)
test_generator = ImageDataGenerator()
train_generator = train_generator.flow(feature_train, target_train, batch_size=64)
test_generator = test_generator.flow(feature_test, target_test, batch_size=64)

# Now perform the training
# `fit_generator` assumes there is an underlying function that provides /
# generates the data into the training process
# ALSO `fit` will run the training in RAM where the generator is MUCH more
# efficient with its memory usage
model.fit_generator(
    train_generator,
    # Integer division over the batch size of the generator
    steps_per_epoch=60_000 // 64,
    epochs=2,
    validation_data=test_generator,
    validation_steps=10_000 // 64,
)
score = model.evaluate(feature_test, target_test)
print(f"Test accuracy: {score[1]:0.2f}")
"""
Epoch 1/2
937/937 [==============================] - 35s 36ms/step - loss: 0.1283 - accuracy: 0.9598 - val_loss: 0.0343 - val_accuracy: 0.9892
Epoch 2/2
937/937 [==============================] - 37s 40ms/step - loss: 0.0575 - accuracy: 0.9822 - val_loss: 0.0246 - val_accuracy: 0.9916
313/313 [==============================] - 2s 6ms/step - loss: 0.0246 - accuracy: 0.9916
Test accuracy: 0.99
"""
