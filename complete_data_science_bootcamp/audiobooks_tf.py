import datetime
from pathlib import Path
import os
import numpy as np
from sklearn import preprocessing
import tensorflow as tf

# Create an ML model that will predict whether a customer will buy again.
# Therefore we will not spend advertising on individuals who are NOT likely
# to come back


AUDIOBOOK_DATA_CSV = "Audiobooks_data.csv"
AUDIOBOOK_TEST_NPZ = "Audiobooks_data_test.npz"
AUDIOBOOK_TRAIN_NPZ = "Audiobooks_data_train.npz"
AUDIOBOOK_VALIDATION_NPZ = "Audiobooks_data_validation.npz"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIOBOOK_DATA_DIR = os.sep.join([CURRENT_DIR, "audiobook-data"])


def npz_file_path(file_name: str) -> str:
    return os.sep.join([AUDIOBOOK_DATA_DIR, file_name])


def file_exists(file_path: str) -> bool:
    return Path(file_path).is_file()


def splits_exist() -> bool:
    # Check if all 3 files exist
    audiobook_test_data = npz_file_path(AUDIOBOOK_TEST_NPZ)
    audiobook_train_data = npz_file_path(AUDIOBOOK_TRAIN_NPZ)
    audiobook_validation_data = npz_file_path(AUDIOBOOK_VALIDATION_NPZ)
    return (
        file_exists(audiobook_test_data)
        and file_exists(audiobook_train_data)
        and file_exists(audiobook_validation_data)
    )


def create_splits():
    data_file_path = os.sep.join([AUDIOBOOK_DATA_DIR, AUDIOBOOK_DATA_CSV])
    if not file_exists(data_file_path):
        raise RuntimeError(f"Unable to find audiobook data file: {data_file_path}")

    raw_csv_data = np.loadtxt(data_file_path, delimiter=",")
    # Data comes thru in the format:
    #
    # 00994,1620,1620,19.73,19.73,1,10.00,0.99,1603.80,5,92,0
    # [Customer_ID], [Book_len_mins (all purchases)], [Book_len_mins_avg], [price], [price_avg], [review], [review_of_10], [completion_pct], [minutes_listened], [support_requests], [days_since_last_interaction], [targets]
    #
    # The larger the days_since_last_interaction the better since they use the platform more frequently
    # If review is not present we update the `review_of_10` column with the average review score
    #
    # Data is collected on a 2 year timeframe and the target indicates if the
    # user has bought another book in the following 6 months after the 2 year
    # collection period

    # Initial probability of picking a result of some class from a dataset is called a prior
    # Priors are balanced with the dataset has equal representation of each of the classes
    # from the start

    # To balance this dataset we make sure that we have an equal number of targets that
    # followed up with a purchase to ones that did not

    unscaled_inputs_all = raw_csv_data[:, 1:-1]
    targets_all = raw_csv_data[:, -1]

    print(raw_csv_data.shape)
    print(unscaled_inputs_all.shape)
    print(targets_all.shape)

    # Shuffle the data before balancing to remove any day effects, etc.
    shuffled_indices = np.arange(unscaled_inputs_all.shape[0])
    np.random.shuffle(shuffled_indices)

    unscaled_inputs_all = unscaled_inputs_all[shuffled_indices]
    targets_all = targets_all[shuffled_indices]

    # Balance the dataset
    num_one_targets = int(np.sum(targets_all))

    zero_targets_counter = 0
    indices_to_remove = []

    for i in range(targets_all.shape[0]):
        if targets_all[i] == 0:
            zero_targets_counter += 1
            if zero_targets_counter > num_one_targets:
                indices_to_remove.append(i)

    print(unscaled_inputs_all.shape[0] - len(indices_to_remove))

    unscaled_inputs_equal_priors = np.delete(
        unscaled_inputs_all, indices_to_remove, axis=0
    )
    targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)

    print(unscaled_inputs_equal_priors.shape)
    print(targets_equal_priors.shape)

    # Standardize the input
    scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)
    print(scaled_inputs.shape)

    # Shuffle the data after balancing to ensure targets that are `1s` will NOT just
    # be in the train_targets
    shuffled_indices = np.arange(scaled_inputs.shape[0])
    np.random.shuffle(shuffled_indices)

    shuffled_inputs = scaled_inputs[shuffled_indices]
    shuffled_targets = targets_equal_priors[shuffled_indices]

    print(shuffled_inputs.shape)
    print(shuffled_targets.shape)

    # Split the dataset into train, validation, and test
    samples_count = shuffled_inputs.shape[0]

    train_samples_count = int(0.8 * samples_count)
    validation_samples_count = int(0.1 * samples_count)
    test_samples_count = samples_count - train_samples_count - validation_samples_count
    print(
        samples_count, train_samples_count, validation_samples_count, test_samples_count
    )

    train_inputs = shuffled_inputs[:train_samples_count]
    train_targets = shuffled_targets[:train_samples_count]

    validation_inputs = shuffled_inputs[
        train_samples_count : train_samples_count + validation_samples_count
    ]
    validation_targets = shuffled_targets[
        train_samples_count : train_samples_count + validation_samples_count
    ]

    test_inputs = shuffled_inputs[train_samples_count + validation_samples_count :]
    test_targets = shuffled_targets[train_samples_count + validation_samples_count :]

    print(
        np.sum(train_targets),
        train_samples_count,
        np.sum(train_targets) / train_samples_count,
    )
    print(
        np.sum(validation_targets),
        validation_samples_count,
        np.sum(validation_targets) / validation_samples_count,
    )
    print(
        np.sum(test_targets),
        test_samples_count,
        np.sum(test_targets) / test_samples_count,
    )

    np.savez(
        npz_file_path(AUDIOBOOK_TEST_NPZ).split(".")[0],
        input=test_inputs,
        targets=test_targets,
    )
    np.savez(
        npz_file_path(AUDIOBOOK_TRAIN_NPZ).split(".")[0],
        input=train_inputs,
        targets=train_targets,
    )
    np.savez(
        npz_file_path(AUDIOBOOK_VALIDATION_NPZ).split(".")[0],
        input=validation_inputs,
        targets=validation_targets,
    )


if not splits_exist():
    create_splits()

# Get the data from the files
train = np.load(npz_file_path(AUDIOBOOK_TRAIN_NPZ))
train_inputs = train["input"].astype(np.float64)
train_targets = train["targets"].astype(np.int64)

validation = np.load(npz_file_path(AUDIOBOOK_VALIDATION_NPZ))
validation_inputs = validation["input"].astype(np.float64)
validation_targets = validation["targets"].astype(np.int64)

test = np.load(npz_file_path(AUDIOBOOK_TEST_NPZ))
test_inputs = test["input"].astype(np.float64)
test_targets = test["targets"].astype(np.int64)

# Create the model
INPUT_SIZE = 10
OUTPUT_SIZE = 2
HIDDEN_LAYER_SIZE = 50

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(HIDDEN_LAYER_SIZE, activation="relu"),
        tf.keras.layers.Dense(HIDDEN_LAYER_SIZE, activation="relu"),
        tf.keras.layers.Dense(OUTPUT_SIZE, activation="softmax"),
    ]
)

# Choose optimizer and loss function
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Training
NUM_EPOCHS = 100
BATCH_SIZE = 100
# Early stopping prevents overfitting by checking if the validation accuracy
# increases in a given epoch. Patience defines how many consecutive increases
# we can tolerate
EARLY_STOPPING = tf.keras.callbacks.EarlyStopping(patience=2)

start = datetime.datetime.now()

model.fit(
    train_inputs,
    train_targets,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    callbacks=[EARLY_STOPPING],
    validation_data=(validation_inputs, validation_targets),
    verbose=2,
)

end = datetime.datetime.now()

print(f"Took {end - start} to complete model training")
# Only made it thru 10 epochs due to stopping early
# Took 0:00:01.846788 to complete model training

# Test the model
test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)
print(f"Test loss: {test_loss:1.2f} Test accuracy: {(100*test_accuracy):1.4f} %")
# Test loss: 0.36 Test accuracy: 81.6964 %
