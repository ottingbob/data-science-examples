import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# `cluster_std` will define how much variation there is between the
# points associated with a given cluster. Smaller values are clustered
# closer together where a larger value makes it less obvious
x, y = make_blobs(n_samples=100, centers=5, random_state=0, cluster_std=3)

print(x.shape)
# The x, y coordinates of the point belonging to a respective cluster
# (100, 2)
print(y.shape)
# The integer cluster label
# (100,)

plt.scatter(x[:, 0], x[:, 1], s=50)

model = KMeans(5)
model.fit(x)

predictions = model.predict(x)

plt.scatter(x[:, 0], x[:, 1], c=predictions, s=50, cmap="rainbow")

plt.show()
