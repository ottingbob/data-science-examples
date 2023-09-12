import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

CREDIT_CSV = "mldl_bootcamp_resources/Datasets/Datasets/credit_data.csv"

credit_data = pd.read_csv(CREDIT_CSV)

print(credit_data.describe())
print(credit_data.corr())

features = credit_data[["income", "age", "loan"]]
target = credit_data.default

# 30% of dataset is for testing
# 70% of dataset is for training
(
    feature_train,
    feature_test,
    target_train,
    target_test,
) = train_test_split(features, target, test_size=0.3)

# Train the logistic regression model on the train dataset
model = LogisticRegression()
model.fit = model.fit(feature_train, target_train)

# Feed the features of the test dataset to the model to then get a grasp
# of the model accuracy
predictions = model.fit.predict(feature_test)

# The confusion matrix describes the performance of a classification model
# The diagonal elements are the correct classifications
# The off-diagonals are the incorrect predictions
print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))

print(model.fit.coef_)
print(model.fit.intercept_)
