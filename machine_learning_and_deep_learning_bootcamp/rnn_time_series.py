import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

TIME_SERIES_CSV = "mldl_bootcamp_resources/PythonMachineLearning (4)/machine_learning (actual one)/daily_min_temperatures.csv"

# This is the window that we will use for observing previous values to be able
# to make predictions to help train our model at time t+1
NUM_OF_PREV_ITEMS = 5

# Make sure the results will be the same every time we run the algorithm
np.random.seed(1)

# Deal with the temperature column exclusively at index `1`
data_frame = pd.read_csv(TIME_SERIES_CSV, usecols=[1])


def plot_data():
    # Plot the dataset
    #
    # We see that the data is stationary -- or the statistical properties are
    # approximately constant
    # The mean and variance are approximately constant
    #
    # These are crucial properties because as a rule of thumb non-stationary time
    # series cannot be predicted in the future
    plt.plot(data_frame)
    plt.show()


# Get the temperature column
data = data_frame.values
# Make sure we are dealing with floating point values
data = data.astype("float32")

# Transform the dataset with mix-max normalization
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# Split the dataset into test and train sets (70 / 30)
train_split = int(len(data) * 0.7)
train = data[0:train_split, :]
test = data[train_split : len(data), :]


# Convert an array of values into a matrix of features
# that are the previous time series values in the past
def reconstruct_data(data_set, n=1):
    x, y = [], []
    for i in range(len(data_set) - n - 1):
        a = data_set[i : (i + n), 0]
        x.append(a)
        y.append(data_set[i + n, 0])

    return np.array(x), np.array(y)


# Create the training and test data matrix
train_x, train_y = reconstruct_data(train, NUM_OF_PREV_ITEMS)
test_x, test_y = reconstruct_data(test, NUM_OF_PREV_ITEMS)

print((train_x.shape[0], 1, train_x.shape[1]))
# (2549, 1, 5)

# In order to use the LSTM (Long-short term memory) architecture:
# Reshape the input to be 3D array: [num_samples, time_steps, num_features]
# time_steps is because we want to predict the next value (t+1)
train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(
    LSTM(
        # Number of neurons
        units=100,
        # Instead of returning a single value return all values for input
        # into the next layer
        return_sequences=True,
        input_shape=(1, NUM_OF_PREV_ITEMS),
    )
)
# Dropout regularization to omit 50% of neurons to avoid overfitting
model.add(Dropout(0.5))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.3))
# This is our last layer so we do not include any sequences to return
model.add(LSTM(units=50))
model.add(Dropout(0.3))
# Now we have a densely connected neural net layer with a single unit
# since we just want a single value -- essentially making it a regression
# problem
model.add(Dense(units=1))

# Optimize the model with ADAM optimizer
model.compile(loss="mean_squared_error", optimizer="adam")
# epochs=10 - run thru the dataset 10 times
# batch_size=16 - update the model params after every 16 data points in
#   the training dataset
model.fit(train_x, train_y, epochs=10, batch_size=16, verbose=2)

# Make predictions and mix-max normalization
test_predict = model.predict(test_x)
test_predict = scaler.inverse_transform(test_predict)
test_labels = scaler.inverse_transform([test_y])

# Evaluate the model
test_score = mean_squared_error(test_labels[0], test_predict[:, 0])
print(f"Score on test set using MSE: {test_score:0.2f}")

# Plot the results (original data + predictions)
#
# empty_like will create an array with the same dimensions as the
# provided array filled with random values
test_predict_plot = np.empty_like(data)
# Reinitialize the values to be invalid values
test_predict_plot[:, :] = np.nan
test_predict_plot[
    len(train_x) + 2 * NUM_OF_PREV_ITEMS + 1 : len(data) - 1, :
] = test_predict

# We apply `inverse_transform` since the dataset has been normalized
# so we get back the original values
plt.plot(scaler.inverse_transform(data))
plt.plot(test_predict_plot, color="green")
plt.show()
