import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

iris_data = datasets.load_iris()

features = iris_data.data
target = iris_data.target

# With grid search we can find optimal parameters with tuning
param_grid = {
    "max_depth": np.arange(1, 10),
}

# 20% of dataset is for testing
# 80% of dataset is for training
(
    feature_train,
    feature_test,
    target_train,
    target_test,
) = train_test_split(features, target, test_size=0.2)

# In every iteration we split the data randomly in cross validation +
# DecisionTreeClassifier so it will initialize the tree randomly. This
# is why we receive different results
tree = GridSearchCV(DecisionTreeClassifier(), param_grid)

tree.fit(feature_train, target_train)

print("Best parameter with Grid Search:", tree.best_params_)
grid_predictions = tree.predict(feature_test)
print(confusion_matrix(target_test, grid_predictions))
print(accuracy_score(target_test, grid_predictions))
