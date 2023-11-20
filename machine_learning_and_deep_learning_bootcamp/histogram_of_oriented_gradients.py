import matplotlib.pyplot as plt
from skimage import data, feature

# Fetch an image of an astronaut
image = data.astronaut()
print(image.shape)
# (512, 512, 3)

# `orientations` defines the number of bins in the histogram
# `pixels_per_cell` defines the size of the patch to use across the image when
#   computing the gradients
# `cells_per_block` deals with normalization so we will apply L2 normalization on
#   2x2 patches which in this case would make it 16px/16px blocks
hog_vector, hog_image = feature.hog(
    image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2",
    visualize=True,
    # Define which axis (part of the dimension of the image) has the RGB channel
    # associated with it
    channel_axis=2,
)

# Single row and 2 columns
figure, axes = plt.subplots(
    1, 2, figsize=(10, 8), subplot_kw=dict(xticks=[], yticks=[])
)
# Plot the first image
axes[0].imshow(image)
axes[0].set_title("Original Image")
# Plot the HOG related image
axes[1].imshow(hog_image)
axes[1].set_title("HOG Image")
plt.show()
