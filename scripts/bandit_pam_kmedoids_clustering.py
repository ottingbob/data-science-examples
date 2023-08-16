from banditpam import KMedoids
import numpy as np
import matplotlib.pyplot as plt

# Generate data from a Gaussian Mixture Model with the given means:
np.random.seed(8080)
n_per_cluster = 40
# means shape is 3 x 2
means = np.array([[0, 0], [-5, 5], [5, 5]])
X = np.vstack([np.random.randn(n_per_cluster, 2) + mu for mu in means])

# fit the data with BanditPam
k_medoids = KMedoids(n_medoids=means.shape[0], algorithm="BanditPAM")
k_medoids.fit(X, "L2")

print(k_medoids.average_loss)
# 1.284395694732666
print(k_medoids.labels)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#   0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#   1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#   2 2 2 2 2 2 2 2 2]

# Visualize the data and the medoids:
for p_idx, point in enumerate(X):
    if p_idx in map(int, k_medoids.medoids):
        plt.scatter(X[p_idx, 0], X[p_idx, 1], color="red", s=40)
    else:
        plt.scatter(X[p_idx, 0], X[p_idx, 1], color="blue", s=10)

plt.show()
