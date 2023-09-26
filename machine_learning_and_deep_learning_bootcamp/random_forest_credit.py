import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

CREDIT_CSV = "mldl_bootcamp_resources/Datasets/Datasets/credit_data.csv"

credit_data = pd.read_csv(CREDIT_CSV)

features = credit_data[["income", "age", "loan"]]
target = credit_data.default

# Machine learning handles arrays and not data-frames
# We reshape the data into 3 columns and a sample row per column
X = np.array(features).reshape(-1, 3)
y = np.array(target)


model = RandomForestClassifier()
predicted = cross_validate(model, X, y, cv=10)
print(np.mean(predicted["test_score"]))
