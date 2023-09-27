from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

digits_data = datasets.load_digits()

features = digits_data.images.reshape((len(digits_data.images), -1))
target = digits_data.target

# With grid search we can find optimal parameters with tuning
param_grid = {
    "n_estimators": [10, 100, 500, 1000],
    "max_depth": [1, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 10, 15, 30, 50],
}

# 20% of dataset is for testing
# 80% of dataset is for training
(
    feature_train,
    feature_test,
    target_train,
    target_test,
) = train_test_split(features, target, test_size=0.2)

random_forest_model = GridSearchCV(
    # Number of jobs to run in parallel (use all cores)
    estimator=RandomForestClassifier(n_jobs=-1, max_features="sqrt"),
    param_grid=param_grid,
    cv=10,
)

random_forest_model.fit(feature_train, target_train)

print("Best parameter with Grid Search:", random_forest_model.best_params_)
# Best parameter with Grid Search:
# {'max_depth': 15, 'min_samples_leaf': 1, 'n_estimators': 1000}

grid_predictions = random_forest_model.predict(feature_test)
print(confusion_matrix(target_test, grid_predictions))
print(accuracy_score(target_test, grid_predictions))
