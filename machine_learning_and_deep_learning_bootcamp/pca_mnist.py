import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mnist_data = datasets.fetch_openml("mnist_784")

features = mnist_data.data
targets = mnist_data.target

print(features.shape)
# (70000, 784)

# 15% of dataset is for testing
# 85% of dataset is for training
(
    feature_train,
    feature_test,
    target_train,
    target_test,
) = train_test_split(features, targets, test_size=0.15)

# Use Z-Scaler standardization on the dataset
scaler = StandardScaler()
scaler.fit(feature_train)

train_img = scaler.transform(feature_train)
test_img = scaler.transform(feature_test)

# We keep 95% of the variance -- so maintain a number of features that
# still represent 95% of the original information
pca = PCA(0.95)
pca.fit(train_img)

train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

print(train_img.shape)
# (59500, 328)
