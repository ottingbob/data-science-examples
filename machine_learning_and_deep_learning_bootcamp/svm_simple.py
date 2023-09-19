import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import plot_decision_regions
from sklearn import svm

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
# 0: blue class
# 1: red class
y = np.concatenate([[0] * 6, [1] * 6])

plt.plot(x_blue, y_blue, "ro", color="blue")
plt.plot(x_red, y_red, "ro", color="red")
plt.plot(2.5, 4.5, "ro", color="green", markersize=15)
# plt.show()

# Hyper-parameter explanation:
#   gamma -> defines how far the influence of a single training example reaches
#       Low value: influence reaches far
#       High value: influence reaches close
#
#   C: Cost Parameter
#   C -> trades off hyperplane surface simplicity + training examples
#        mis-classifications
#       Low value: simple / smooth hyperplane surface (underfitting)
#       High value: all training examples classified correctly but complex
#           surface (overfitting)
#
classifier = svm.SVC()
classifier.fit(X, y)

print(classifier.predict([[2.5, 4.5]]))

# Plot the decision boundary
plot_decision_regions(X, y, clf=classifier, legend=2)

plt.axis([-0.5, 10, -0.5, 10])
plt.show()
