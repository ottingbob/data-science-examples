import cv2
import numpy as np

IMAGE_FILE = "mldl_bootcamp_resources/PythonMachineLearning (4)/computer_vision_course_materials/camus.jpg"
COLOR_IMAGE_FILE = "mldl_bootcamp_resources/PythonMachineLearning (4)/computer_vision_course_materials/bird.jpg"

# `0` indicates we are going to handle a grayscale image
# image = cv2.imread(IMAGE_FILE, 0)
# Or just do this which I think reads better
image = cv2.imread(IMAGE_FILE, cv2.IMREAD_GRAYSCALE)


def display_image():
    cv2.imshow("Computer Vision", image)
    # Wait for the image to be closed
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Values closer to 0 are darker pixels
# Values closer to 255 are brighter pixels
print(image)
print(image.shape)
# (545, 800)

color_image = cv2.imread(COLOR_IMAGE_FILE, cv2.IMREAD_COLOR)
# So this will have 3 values [0, 255] for every pixel - Red Green Blue
# 24 bits (3 bytes) for the RGB components of the image
print(color_image)
print(color_image.shape)
# (400, 600, 3)

# We store the RGB values on 8 bits
# 2 ^ 8 = 256
#
# Find the maximum pixel intensity value in this 3D array
print(np.amax(color_image))
# 255
