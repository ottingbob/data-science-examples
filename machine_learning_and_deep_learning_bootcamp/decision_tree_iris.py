import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.tree import DecisionTreeClassifier

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

# Default value is gini index but we define our criterion to be `entropy`
# and information gain
model = DecisionTreeClassifier(criterion="entropy")
# model = DecisionTreeClassifier(criterion="gini")

predicted = cross_validate(model, features, target, cv=10)
print(np.mean(predicted["test_score"]))
