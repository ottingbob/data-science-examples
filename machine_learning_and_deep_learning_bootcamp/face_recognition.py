import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

olivetti_data = datasets.fetch_olivetti_faces()

# There are 400 images
# 10x40: 40 people - 1 person has 10 images
# 1 image - 64x64 pixels
features = olivetti_data.data
targets = olivetti_data.target

# Pixel intensity values are already normalized

print(features.shape)
# (400, 4096)
print(targets.shape)
# (400,)


def plot_faces():
    # Plot and visualize the dataset
    fig, sub_plots = plt.subplots(nrows=5, ncols=8, figsize=(14, 8))
    # Put all the subplots into the same figure
    sub_plots = sub_plots.flatten()

    for unique_user_id in np.unique(targets):
        # Track id of a given image in the array
        image_index = unique_user_id * 8
        current_plot = sub_plots[unique_user_id]
        current_plot.imshow(features[image_index].reshape(64, 64), cmap="gray")
        current_plot.set_xticks([])
        current_plot.set_yticks([])
        current_plot.set_title(f"Face ID: {unique_user_id}")

    plt.suptitle("The dataset (40 people)")
    plt.show()


def plot_face_id_0():
    # Plot the 10 images for the first person (id=0)
    fig, sub_plots = plt.subplots(nrows=2, ncols=5, figsize=(14, 8))
    # Put all the subplots into the same figure
    sub_plots = sub_plots.flatten()

    for j in range(10):
        # Track id of a given image in the array
        # image_index = unique_user_id * 8
        current_plot = sub_plots[j]
        current_plot.imshow(features[j].reshape(64, 64), cmap="gray")
        current_plot.set_xticks([])
        current_plot.set_yticks([])
        current_plot.set_title(f"Face ID 0 #{j}")

    plt.suptitle("The different images of Face ID 0")
    plt.show()


# Split original dataset into training and test data
# 25% of dataset is for testing
# 75% of dataset is for training
# `stratify` means the data is split in such a way to use as class labels
(feature_train, feature_test, target_train, target_test,) = train_test_split(
    features, targets, test_size=0.25, stratify=targets, random_state=0
)


def plot_pca():
    # Use principal component analysis to find optimal number of eigenvectors
    # Turn the 4096 features into 100 and minimize amount of information lost
    # Said differently -- we want to maximize the explained variance
    pca = PCA()
    pca.fit(features)

    plt.figure(1, figsize=(12, 8))
    plt.plot(pca.explained_variance_, linewidth=2)
    plt.xlabel("Components")
    plt.ylabel("Explained Variances")
    plt.show()


# Based on plotting the PCA we can see that `100` components still helps us
# maintain enough variance within the feature data to reduce the dimensionality
# of the dataset by an order of magnitude BUT also allow us to maintain the same
# information to help us categorize the data
#
# `whiten` can help improve the predictive accuracy of the estimators so we set
# this value to `True`
pca = PCA(n_components=100, whiten=True)

# Fit the PCA to the training data
pca.fit(feature_train)
# Apply PCA / dimensionality reduction on the training & test data
feature_train_pca = pca.transform(feature_train)
feature_test_pca = pca.transform(feature_test)


def plot_eigen_faces():
    # After we find the optimal 100 PCA numbers we can check the `eigenfaces`
    # 1 principal component (eigenvector) has 4096 features
    number_of_eigenfaces = len(pca.components_)
    print(number_of_eigenfaces)
    # 100

    # We can represent every face (sample) in the training / test dataset with a
    # linear combination of these eigen-faces (eigenvectors)
    eigen_faces = pca.components_.reshape((number_of_eigenfaces, 64, 64))
    fig, sub_plots = plt.subplots(nrows=10, ncols=10, figsize=(14, 8))
    # Put all the subplots into the same figure
    sub_plots = sub_plots.flatten()

    for j in range(number_of_eigenfaces):
        # Track id of a given image in the array
        # image_index = unique_user_id * 8
        current_plot = sub_plots[j]
        current_plot.imshow(eigen_faces[j].reshape(64, 64), cmap="gray")
        current_plot.set_xticks([])
        current_plot.set_yticks([])
        current_plot.set_title(f"Eigen Face #{j}")

    plt.suptitle("Eigenfaces")
    plt.show()


# We have reduced the number of dimensions on the sample from 4096 to 100
print(feature_train.shape)
# (300, 4096)
print(feature_train_pca.shape)
# (300, 100)

models = [
    ("Logistic Regression", LogisticRegression()),
    ("Support Vector Machines", SVC()),
    ("Naive Bayes Classifier", GaussianNB()),
]
for name, model in models:
    classifier_model = model
    classifier_model.fit(feature_train_pca, target_train)
    target_predictions = classifier_model.predict(feature_test_pca)
    accuracy_score = metrics.accuracy_score(target_test, target_predictions)
    print(f"Results with {name}")
    print(f"Accuracy Score: {accuracy_score}")

# Use K-Fold cross validation. In this case we do not need to split the data
feature_train_pca = pca.transform(feature_train)
print("Now with K-Fold cross validation:")
for name, model in models:
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)
    cv_scores = cross_val_score(model, feature_train_pca, target_train, cv=kfold)
    print(f"Mean of the {name} cross-validation scores: {cv_scores.mean()}")
