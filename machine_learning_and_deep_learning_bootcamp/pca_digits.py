import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

digits_data = datasets.load_digits()

# features = digits_data.images.reshape((len(digits_data.images), -1))
# target = digits_data.target

X_digits = digits_data.data
y_digits = digits_data.target

# Transform the 64 features into just 10
# This is also the amount of eigenvectors to use!!
estimator = PCA(n_components=10)
X_pca = estimator.fit_transform(X_digits)

print(X_pca.shape)
# (1797, 2)

print(X_digits.shape)
# (1797, 64)

colors = [
    "black",
    "blue",
    "purple",
    "yellow",
    "white",
    "red",
    "lime",
    "cyan",
    "orange",
    "gray",
]

for i in range(len(colors)):
    px = X_pca[:, 0][y_digits == i]
    py = X_pca[:, 1][y_digits == i]
    plt.scatter(px, py, c=colors[i])
    plt.legend(digits_data.target_names)

plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")

# Explained variance shows how much information can be attributed to the
# principle components
# We should be trying to aim for capturing 95% of the original dataset when
# performing PCA to reduce dimensionality in our dataset
print("Explained variance:", estimator.explained_variance_ratio_)
print("Explained variance sum:", sum(estimator.explained_variance_ratio_))

plt.show()
