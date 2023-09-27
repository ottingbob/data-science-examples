import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

x = np.array([[1, 1], [1.5, 1], [3, 3], [4, 4], [3, 3.5], [3.5, 4]])

plt.scatter(x[:, 0], x[:, 1], s=50)
plt.show()

# `single` defines that the algorithm will calculate the distance between two
# datapoints by using this formula called the farthest point algorithm
# This has quadratic running time so this will be SUPER slow
linkage_matrix = linkage(x, "single")

# `truncate_mode` allows us to omit certain values out of the dendrogram if
# the initial dataset is large
dendrogram = dendrogram(linkage_matrix, truncate_mode="none")

plt.title("Hierarchical Clustering")
plt.show()
