from sklearn import datasets, svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

iris_data = datasets.load_iris()

features = iris_data.data
target = iris_data.target

# 30% of dataset is for testing
# 70% of dataset is for training
(
    feature_train,
    feature_test,
    target_train,
    target_test,
) = train_test_split(features, target, test_size=0.3)

model = svm.SVC()

# Define possible values to test out for the related parameters
# There are 12 x 4 x 3 = 144 combinations to test here
param_grid = {
    "C": [0.1, 1, 5, 10, 20, 30, 40, 50, 60, 70, 100, 200],
    "gamma": [1, 0.1, 0.01, 0.001],
    "kernel": ["rbf", "poly", "sigmoid"],
}

grid = GridSearchCV(model, param_grid, refit=True)
grid.fit(feature_train, target_train)

# This will change based on the train / test data given to the model
print(grid.best_estimator_)

grid_predictions = grid.predict(feature_test)

print(confusion_matrix(target_test, grid_predictions))
print(accuracy_score(target_test, grid_predictions))
