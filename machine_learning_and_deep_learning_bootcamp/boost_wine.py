import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

WINE_CSV = "mldl_bootcamp_resources/Datasets/Datasets/wine.csv"

wine_data = pd.read_csv(WINE_CSV, sep=";")


# Classify the quality column in the dataset to turn it into a binary
# label to train the dataset on
def is_tasty(quality: int) -> int:
    return 1 if quality >= 7 else 0


feature_cols = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
features = wine_data[feature_cols]
targets = wine_data["quality"].apply(is_tasty)

X = np.array(features).reshape(-1, len(feature_cols))

# Preprocessing actually decreases the accuracy when using AdaBoost...
# Transform feature values into range [0, 1]
# X = preprocessing.MinMaxScaler().fit_transform(X)

y = np.array(targets)

# 20% of dataset is for testing
# 80% of dataset is for training
(
    feature_train,
    feature_test,
    target_train,
    target_test,
) = train_test_split(X, y, test_size=0.2)

param_dist = {
    "n_estimators": [10, 50, 200],
    "learning_rate": [0.01, 0.05, 0.3, 1],
}

grid_search = GridSearchCV(
    estimator=AdaBoostClassifier(),
    param_grid=param_dist,
    cv=10,
)
grid_search.fit(feature_train, target_train)

print("Best parameter with Grid Search:", grid_search.best_params_)
# Best parameter with Grid Search: {'learning_rate': 0.3, 'n_estimators': 200}

grid_predictions = grid_search.predict(feature_test)

print(confusion_matrix(target_test, grid_predictions))
print(accuracy_score(target_test, grid_predictions))
