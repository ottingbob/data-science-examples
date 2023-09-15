import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier

CREDIT_CSV = "mldl_bootcamp_resources/Datasets/Datasets/credit_data.csv"

credit_data = pd.read_csv(CREDIT_CSV)

features = credit_data[["income", "age", "loan"]]
target = credit_data.default

# Machine learning handles arrays, not data-frames
# We reshape the 3 features into a row per sample with the 3 columns
X = np.array(features).reshape(-1, 3)
y = np.array(target)

# Apply min-max transform on the feature values
#
# This allows us to normalize the values before running the kNN algorithm
# such that the distance measurements are not dominated by the larger values
#
# Min-max transforms a feature such that all the values will fall in
# the range [0, 1]
X = preprocessing.MinMaxScaler().fit_transform(X)

# 30% of dataset is for testing
# 70% of dataset is for training
(
    feature_train,
    feature_test,
    target_train,
    target_test,
) = train_test_split(X, y, test_size=0.3)

# Use the 20 closest neighbors on how to classify the given item
model = KNeighborsClassifier(n_neighbors=20)
fitted_model = model.fit(feature_train, target_train)
predictions = fitted_model.predict(feature_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))

# Use cross-validation to find the optimal number of `n_neighbors`
cross_valid_scores = []
for k in range(1, 100):
    knn = KNeighborsClassifier(n_neighbors=k)
    # We use `cv` to indicate 10 folds when running our cross validation
    scores = cross_val_score(knn, X, y, cv=10, scoring="accuracy")
    cross_valid_scores.append(scores.mean())

print("Optimal k with cross-validation:", np.argmax(cross_valid_scores))
# Optimal k with cross-validation: 32

model = KNeighborsClassifier(n_neighbors=32)
fitted_model = model.fit(feature_train, target_train)
predictions = fitted_model.predict(feature_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))
