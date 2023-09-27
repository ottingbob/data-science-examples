import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

SHOPPING_CSV = "mldl_bootcamp_resources/PythonMachineLearning (4)/Datasets/Datasets/shopping_data.csv"

shopping_data = pd.read_csv(SHOPPING_CSV)

# We use `iloc` to get the values in the annual income and spending score
# columns for every row in the dataset
data = shopping_data.iloc[:, 3:5].values

# The features we have are annual salary and shopping score
print(data)
plt.figure(figsize=(10, 7))
plt.title("Market Segmentation Dendrogram")

# Construct the dendrogram from the linkage matrix
dendrogram = dendrogram(linkage(data, method="ward"))

# We can use the dendrogram to find out the optimal value of the `k` number
# of clusters
# 1) Determine the largest vertical distance that does not intersect any of
#   the other clusters
# 2) Draw a horizontal line at the top and at the bottom (at both extremities)
# 3) Count the number of vertical lines going through the horizontal line:
#   that is the optimal `k` number of clusters

cluster = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="ward")
cluster.fit_predict(data)

plt.figure(figsize=(10, 7))
plt.scatter(data[:, 0], data[:, 1], c=cluster.labels_, cmap="rainbow")
plt.title("Market Segmentation")
plt.xlabel("Income")
plt.ylabel("Affinity / Spending Score")
plt.show()
