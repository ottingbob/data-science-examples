import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

CREDIT_CSV = "mldl_bootcamp_resources/Datasets/Datasets/credit_data.csv"

credit_data = pd.read_csv(CREDIT_CSV)

features = credit_data[["income", "age", "loan"]]
target = credit_data.default

# Machine learning handles arrays, not data-frames
# We reshape the 3 features into a row per sample with the 3 columns
X = np.array(features).reshape(-1, 3)
y = np.array(target)

# 30% of dataset is for testing
# 70% of dataset is for training
(
    feature_train,
    feature_test,
    target_train,
    target_test,
) = train_test_split(X, y, test_size=0.3)

# Naive since it assumes that the features are independent of each other
# even though in this case (and likely most others) they are not
model = GaussianNB()
fitted_model = model.fit(feature_train, target_train)
predictions = fitted_model.predict(feature_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))
