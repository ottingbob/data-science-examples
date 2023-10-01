import os

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from PIL import Image

# The directory that has the training data images
training_dir = "mldl_bootcamp_resources/Datasets/Datasets/smiles_dataset/training_set/"

# The directory that has the testing data images
testing_dir = "mldl_bootcamp_resources/Datasets/Datasets/smiles_dataset/test_set/"

# Where we will hold the pixel data for the given smile images
pixel_intensities = []

happy_label = [1, 0]
sad_label = [0, 1]

# Use one-hot-encoding:
# happy (1,0)
# sad (1,0)
labels = []

for filename in os.listdir(training_dir):
    # `convert` with `1` will drop RGB values and covert the image to
    # grayscale meaning that the pixel will go from 0 (black) to 255 (white)
    image = Image.open(training_dir + filename).convert("1")
    pixel_intensities.append(list(image.getdata()))
    if filename[0:5] == "happy":
        labels.append(happy_label)
    elif filename[0:3] == "sad":
        labels.append(sad_label)
    else:
        print("Unable to classify filename with a known label: ", filename)

pixel_intensities = np.array(pixel_intensities)
# 32x32 images so we will have 1024 items in a given subarray
print(pixel_intensities.shape)
# (20, 1024)

labels = np.array(labels)
print(labels.shape)
# (20, 2)

# Apply min-max normalization (here just /255)
pixel_intensities = pixel_intensities / 255.0
print(pixel_intensities)

# Create the deep neural network model
model = Sequential()
model.add(Dense(1024, input_dim=1024, activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(2, activation="softmax"))

optimizer = Adam(learning_rate=0.005)

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    # When we use a `softmax` activation function in our output layer we need
    # to use binary cross entropy as our metric
    metrics=["accuracy"],
)

model.fit(pixel_intensities, labels, epochs=1_000, batch_size=20, verbose=2)

test_pixels = []
test_labels = []

for filename in os.listdir(testing_dir):
    # `convert` with `1` will drop RGB values and covert the image to
    # grayscale meaning that the pixel will go from 0 (black) to 255 (white)
    image = Image.open(testing_dir + filename).convert("1")
    test_pixels.append(list(image.getdata()))
    if filename[0:5] == "happy":
        test_labels.append(happy_label)
    elif filename[0:3] == "sad":
        test_labels.append(sad_label)
    else:
        print("Unable to classify filename with a known label: ", filename)


test_pixels = np.array(test_pixels)
test_pixels = test_pixels / 255.0
test_labels = np.array(test_labels)

# Multiprocessing allows us to boost up the algorithm
results = model.evaluate(test_pixels, test_labels, use_multiprocessing=True)

# Print the metrics associated with the model training
print("Training is finished. The loss and accuracy values are:")
print(results)

print("And the predictions:")
print(model.predict(test_pixels).round())
print("Expected: ", test_labels)
