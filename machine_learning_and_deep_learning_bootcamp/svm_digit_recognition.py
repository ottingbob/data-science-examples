import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.metrics import accuracy_score, confusion_matrix

# from sklearn.model_selection import GridSearchCV, train_test_split

digit_data = datasets.load_digits()

# Will have 8 columns and 8 rows since we are dealing with an
# 8x8 pixel image
features = digit_data.images
target = digit_data.target

images_and_labels = list(zip(digit_data.images, digit_data.target))


def plot_sample_images():
    # Plot some sample images to get an idea on what we are working with
    for index, (image, label) in enumerate(images_and_labels[:6]):
        plt.subplot(2, 3, index + 1)
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        plt.title(f"Target: {label}")

    plt.show()


# To apply a classifier on this data we need to flatten the image
# Instead of an 8x8 matrix we have to use a one-dimensional array with
# 64 items
data = features.reshape((len(features), -1))

classifier = svm.SVC(gamma=0.001)

# Manually create train / test split with 75% of original dataset
# for training
train_test_split = int(len(features) * 0.75)

classifier.fit(data[:train_test_split], target[:train_test_split])

# Now predict the value on the remaining 25%
expected = target[train_test_split:]
predicted = classifier.predict(data[train_test_split:])

print(confusion_matrix(expected, predicted))
print(accuracy_score(expected, predicted))
