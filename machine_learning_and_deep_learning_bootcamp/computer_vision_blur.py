import cv2
import numpy as np

COLOR_IMAGE_FILE = "mldl_bootcamp_resources/PythonMachineLearning (4)/computer_vision_course_materials/bird.jpg"

original_image = cv2.imread(COLOR_IMAGE_FILE, cv2.IMREAD_COLOR)

# Create a 5x5 array of `1` and normalize it with the number
# of items (5x5 = 25)
blur_kernel = np.ones((5, 5)) / 25

# `-1` defines the destination depth meaning that the depth
# will be the same as the original image
blur_image = cv2.filter2D(original_image, -1, blur_kernel)

# Gaussian blur is used to reduce noise !!

cv2.imshow("Original Image", original_image)
cv2.imshow("Blurred Image", blur_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
