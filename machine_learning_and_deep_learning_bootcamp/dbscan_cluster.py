import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN, KMeans

# Make two interleaving half circles
X, y = datasets.make_moons(n_samples=1500, noise=0.05)

print(X.shape)
# (1500, 2)

x1 = X[:, 0]
x2 = X[:, 1]

# DBSCAN - density based spatial cluster of applications with noise
# `eps` epsilon value defines the maximum distance between two samples to
# be considered as part of the same cluster
dbscan = DBSCAN(eps=0.1)
dbscan.fit(X)


# Contains the labels for the points that belong to the respective cluster
y_pred = dbscan.labels_.astype(int)
print(y_pred)

colors = np.array(["#ff0000", "#00ff00"])

plt.scatter(x1, x2, s=5, c=colors[y_pred])
plt.show()

# Results with K-Means Clustering
# This will end up misclassifying some of the data points in each cluster
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_pred = kmeans.labels_.astype(int)

plt.scatter(x1, x2, s=5, c=colors[y_pred])
plt.show()
