import cv2
import numpy as np

UNSHARP_IMAGE_FILE = "mldl_bootcamp_resources/PythonMachineLearning (4)/computer_vision_course_materials/unsharp_bird.jpg"

original_image = cv2.imread(UNSHARP_IMAGE_FILE, cv2.IMREAD_COLOR)

# Sharpen kernel
sharpen_kernel = np.array(
    [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ]
)

# `-1` defines the destination depth meaning that the depth
# will be the same as the original image
# This means the filter result will have the same dimension
# as the input image
sharpen_image = cv2.filter2D(original_image, -1, sharpen_kernel)

# Show the results
cv2.imshow("Original Image", original_image)
cv2.imshow("Sharpen Image", sharpen_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
