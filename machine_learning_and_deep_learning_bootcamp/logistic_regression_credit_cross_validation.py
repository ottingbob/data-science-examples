import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_validate

CREDIT_CSV = "mldl_bootcamp_resources/Datasets/Datasets/credit_data.csv"

credit_data = pd.read_csv(CREDIT_CSV)

features = credit_data[["income", "age", "loan"]]
target = credit_data.default

# Machine learning handles arrays and not data-frames
# We reshape the data into 3 columns and a sample row per column
X = np.array(features).reshape(-1, 3)
y = np.array(target)

# Train the logistic regression model on the train dataset
model = LogisticRegression()
# We set the number of folds to be 5
predicted = cross_validate(model, X, y, cv=5)

# Prints out the accuracy of each fold
print(predicted["test_score"])
print(np.mean(predicted["test_score"]))
