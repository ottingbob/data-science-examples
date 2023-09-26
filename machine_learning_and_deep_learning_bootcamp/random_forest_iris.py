from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

iris_data = datasets.load_iris()

features = iris_data.data
target = iris_data.target

# 20% of dataset is for testing
# 80% of dataset is for training
(
    feature_train,
    feature_test,
    target_train,
    target_test,
) = train_test_split(features, target, test_size=0.2)

# `max_features` defines the number of features to consider when looking
# for the best split
model = RandomForestClassifier(n_estimators=1000, max_features="sqrt")
fitted_model = model.fit(feature_train, target_train)

predictions = fitted_model.predict(feature_test)
print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))
