from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from skimage import data, feature, transform
from skimage.io import imread
from sklearn.datasets import fetch_lfw_people
from sklearn.feature_extraction.image import PatchExtractor
from sklearn.svm import LinearSVC

# We can load a data-set of human faces
# These are the positive samples
human_faces = fetch_lfw_people()
positive_images = human_faces.images[:10_000]

# Each image is 62x47 pixels
print(positive_images.shape)
# (10000, 62, 47)

# Plot an image to see what we are workin with
# plt.imshow(positive_images[0], cmap="gray")
# plt.show()

# Now we will load the non-human faces
# Which in our case will be considered the negative samples
non_face_topics = ["moon", "text", "coins"]
# This will invoke `data.<non_face_topic>()` by metaprogramming
negative_samples = [(getattr(data, name)()) for name in non_face_topics]


# We will use PatchExtractor to generate several variants of these images
def generate_random_samples(
    image, num_of_generated_images=100, patch_size=positive_images[0].shape
):

    extractor = PatchExtractor(
        patch_size=patch_size,
        max_patches=num_of_generated_images,
        random_state=42,
    )
    # `np.newaxis` will increment the dimension of the image array
    # We will create new images (samples) and then store them in this new
    # dimension that we add here
    patches = extractor.transform((image[np.newaxis]))
    return patches


# We will generate 3000 negative samples (negative samples without a human face)
negative_images = np.vstack(
    [generate_random_samples(image, 1000) for image in negative_samples]
)
print(negative_images.shape)
# (3000, 62, 47)


def visualize_generated_images():
    # Visualize 100 negative images
    fig, ax = plt.subplots(10, 10)
    for i, axis in enumerate(ax.flat):
        axis.imshow(negative_images[i], cmap="gray")
        axis.axis("off")
    plt.show()


# We construct the training set with the output variables (labels)
# and we construct the HOG features
# ||-- TIME CONSUMING PROCEDURE --||
X_train = np.array(
    [feature.hog(image) for image in chain(positive_images, negative_images)]
)
# Labels are between 0 and 1
# 1: Face
# 0: Non-Face
y_train = np.zeros(X_train.shape[0])
y_train[: positive_images.shape[0]] = 1

# We can construct the SVM
svm = LinearSVC()
# This is when SVM learns the parameters for the model based on the training
# data set
svm.fit(X_train, y_train)

FACE_1_PNG = "mldl_bootcamp_resources/PythonMachineLearning (4)/computer_vision_course_materials/girl_face.png"
FACE_2_PNG = "mldl_bootcamp_resources/PythonMachineLearning (4)/computer_vision_course_materials/male_face.png"
BIRD_JPG = "mldl_bootcamp_resources/PythonMachineLearning (4)/computer_vision_course_materials/bird.jpg"

# Read the test images
test_pairs = [
    (FACE_1_PNG, 1),
    (FACE_2_PNG, 1),
    (BIRD_JPG, 0),
]
for test_img_path, expected_cls in test_pairs:
    test_image = imread(fname=test_img_path)
    test_image = transform.resize(test_image, positive_images[0].shape)
    test_image_hog = np.array([feature.hog(test_image, channel_axis=2)])
    prediction = svm.predict(test_image_hog)
    print(f"Prediction made by SVM: {prediction} Expected: {expected_cls}")
