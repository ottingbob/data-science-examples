import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

sns.set()


# Load country clusters data
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
COUNTRY_DATA = os.sep.join([CURRENT_DIR, "country-clusters.csv"])
data = pd.read_csv(COUNTRY_DATA)

# Create a 2x2 grid for subplots
plt.subplot(2, 2, 1)  # top left corner
plt.title("Raw datapoints")
plt.scatter(data["Longitude"], data["Latitude"])
# Set limits of the axes to resemble the world map with longitude
# being y and latitude being x
plt.xlim(-180, 180)
plt.ylim(-90, 90)

# Select the features
x = data.iloc[:, 1:3]

# Clustering
kmeans = KMeans(3, n_init="auto")

# Fit the input data and cluster the data in X in K clusters
kmeans.fit(x)

# Predicted clusters for each obseration
identified_clusters = kmeans.fit_predict(x)
print(identified_clusters)

# Create a copy of the data
data_with_clusters = data.copy()

# Create a new series containing the identified cluster for each observation
data_with_clusters["Cluster"] = identified_clusters
print(data_with_clusters)

# Plot the data using longitude and latitude
plt.subplot(2, 2, 2)  # top right corner
plt.title("Clustered by Lat / Long")
plt.scatter(
    data_with_clusters["Longitude"],
    data_with_clusters["Latitude"],
    c=data_with_clusters["Cluster"],
    cmap="rainbow",
)
plt.xlim(-180, 180)
plt.ylim(-90, 90)

# Now we also have an example to cluster WRT categorical data
# which in this case is the language of the country
data_mapped = data.copy()
language_map = {"English": 0, "French": 1, "German": 2}
data_mapped["Language"] = data_mapped["Language"].map(language_map)
print(data_mapped)

x = data_mapped.iloc[:, 3:4]

kmeans = KMeans(3, n_init="auto")
kmeans.fit(x)
identified_clusters = kmeans.fit_predict(x)
data_with_clusters = data_mapped.copy()
data_with_clusters["Cluster"] = identified_clusters

plt.subplot(2, 2, 3)  # bottom left corner
plt.title("Clustered by Language")
plt.scatter(
    data_with_clusters["Longitude"],
    data_with_clusters["Latitude"],
    c=data_with_clusters["Cluster"],
    cmap="rainbow",
)
plt.xlim(-180, 180)
plt.ylim(-90, 90)

# Now we can use both categorical and numerical data in clustering

# Use the elbow method to:
# 1) minimize the distance between points in a cluster
# 2) maximize the distance between clusters
#
# The elbow method essentially optimizes the WCSS (Within-Cluster sum of squares)
# to be as low as possible while still having a cluster size small enough to be
# able to help reason about the data being clustered

x = data_mapped.iloc[:, 1:4]
wcss = []

number_clusters = range(1, 7)
for i in number_clusters:
    kmeans = KMeans(i, n_init="auto")
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.subplot(2, 2, 4)  # bottom right corner
plt.plot(number_clusters, wcss)
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("Within-Cluster Sum of Squares")

plt.tight_layout()
plt.show()
