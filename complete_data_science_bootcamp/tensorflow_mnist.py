import os

import tensorflow as tf

# from tensorflow.python.data.prefetch_op import _PrefetchDataset
from tensorflow.python.data import Dataset

import tensorflow_datasets as tfds


###
# Prepare the data

current_file_dir = os.path.dirname(os.path.abspath(__file__))
mnist_dataset, mnist_info = tfds.load(
    name="mnist", data_dir=current_file_dir, with_info=True, as_supervised=True
)

# Technically this is a `_PrefetchDataset` when printing the type but
# it doesn't look like that type resolves when trying to do the import...
mnist_train: Dataset = mnist_dataset["train"]
mnist_test = mnist_dataset["test"]

num_validation_samples = 0.1 * mnist_info.splits["train"].num_examples
num_validation_samples = tf.cast(num_validation_samples, tf.int64)

num_test_samples = mnist_info.splits["test"].num_examples
num_test_samples = tf.cast(num_test_samples, tf.int64)


# Scale our data to have inputs between 0 and 1
def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.0
    return image, label


scaled_train_and_validation_data = mnist_train.map(scale)
test_data = mnist_test.map(scale)

BUFFER_SIZE = 10_000
BATCH_SIZE = 100

shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(
    BUFFER_SIZE
)
validation_data = shuffled_train_and_validation_data.take(num_validation_samples)
train_data = shuffled_train_and_validation_data.skip(num_validation_samples)

train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(num_validation_samples)
test_data = test_data.batch(num_test_samples)

# TODO: Wonder why we are doing a next / iter here
validation_inputs, validation_targets = next(iter(validation_data))

###
# Outline the model

INPUT_SIZE = 784
OUTPUT_SIZE = 10
HIDDEN_LAYER_SIZE = 50

model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(HIDDEN_LAYER_SIZE, activation="relu"),
        tf.keras.layers.Dense(HIDDEN_LAYER_SIZE, activation="relu"),
        tf.keras.layers.Dense(OUTPUT_SIZE, activation="softmax"),
    ]
)

# Choose the optimizer and the loss function
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Training
NUM_EPOCHS = 5
model.fit(
    train_data,
    epochs=NUM_EPOCHS,
    validation_data=(validation_inputs, validation_targets),
    verbose=2,
)

# Test the model
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test loss: {test_loss:1.2f} Test accuracy: {(100*test_accuracy):1.4f} %")
# Test loss: 0.12 Test accuracy: 96.2800 %
