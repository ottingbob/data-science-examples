import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

x_blue = np.array([0.3, 0.5, 1, 1.4, 1.7, 2])
y_blue = np.array([1, 4.5, 2.3, 1.9, 8.9, 4.1])

x_red = np.array([3.3, 3.5, 4, 4.4, 5.7, 6])
y_red = np.array([7, 1.5, 6.3, 1.9, 2.9, 7.1])

pairs = []
for pair in zip(x_blue, y_blue):
    pairs.append(list(pair))

for pair in zip(x_red, y_red):
    pairs.append(list(pair))

X = np.array(pairs)
y = np.concatenate([[0] * 6, [1] * 6])

plt.plot(x_blue, y_blue, "ro", color="blue")
plt.plot(x_red, y_red, "ro", color="red")
plt.plot(3, 5, "ro", color="green", markersize=15)
plt.axis([-0.5, 10, -0.5, 10])
plt.show()

# There is no underlying model used in this algorithm
classifier = KNeighborsClassifier(n_neighbors=3)
# Calculates the euclidean distances between the features
classifier.fit(X, y)

predict = classifier.predict(np.array([[3, 5]]))
print(predict)
