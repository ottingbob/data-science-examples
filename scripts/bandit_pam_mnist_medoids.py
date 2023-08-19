import os

from banditpam import KMedoids

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tensorflow_datasets as tfds
import tensorflow as tf

###
# Prepare the data

current_file_dir = os.path.dirname(os.path.abspath(__file__))
mnist_dataset, mnist_info = tfds.load(
    name="mnist", data_dir=current_file_dir, with_info=True, as_supervised=True
)
mnist_train = mnist_dataset["train"]

# TODO: Should probably shuffle the data to get a random sampling...

# Apply batching and select the first 1000 records
mnist_train = mnist_train.batch(1000)
# Take the first batch of 1000 records
# print(type(mnist_train))
# mnist_train: tf.python.data.ops.take_op._TakeDataset
mnist_train = mnist_train.take(1)

# Read the local MNIST file and see what we get...
# _X = pd.read_csv("./MNIST_100.csv", sep=",", header=None).to_numpy()
#
# (100, 784)
# print(_X.shape)


def visualize_medoids(X, medoids):
    # Visualize the data and the medoids:
    for p_idx, point in enumerate(X):
        if p_idx in map(int, medoids):
            plt.scatter(X[p_idx, 0], X[p_idx, 1], color="red", s=40)
        else:
            plt.scatter(X[p_idx, 0], X[p_idx, 1], color="blue", s=10)

    plt.savefig("./assets/mnist_kmedoids_clustering.png")
    plt.show()


batch: tf.Tensor
for batch in mnist_train:
    images, labels = batch

    # print(type(images), type(labels))
    # <class 'tensorflow.python.framework.ops.EagerTensor'> <class 'tensorflow.python.framework.ops.EagerTensor'>
    # print(images.shape, labels.shape)
    # (1000, 28, 28, 1) (1000,)
    # print(images.numpy())

    # Reshape the tensor in the batch from a (28, 28, 1) tensor to a
    # flat (784,) tensor converting the 2D images into 1D vectors
    images = tf.reshape(images, (1000, 784))
    images = images.numpy()

    # (1000, 784)
    # print(images.shape)

    # Load 1000-point subset of MNIST and calculate its t-SNE embeddings
    # for visualization
    X_tsne = TSNE(n_components=2).fit_transform(images)

    # Fit the data with BanditPAM
    k_medoids = KMedoids(n_medoids=10, algorithm="BanditPAM")
    k_medoids.fit(images, "L2")

    visualize_medoids(X_tsne, k_medoids.medoids)

    break
