from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
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

# Base estimator is a decision tree classifier with max_depth=1
# `n_estimators` is the number of iterations in the boosting algorithm
# `learning_rate` shrinks the contribution of each classifier by this
#   value. There is trade off between this and number of estimators
boosty = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=1,
    random_state=123,
)
boosty = boosty.fit(feature_train, target_train)

predictions = boosty.predict(feature_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))
