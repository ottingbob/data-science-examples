import cv2
import numpy as np

COLOR_IMAGE_FILE = "mldl_bootcamp_resources/PythonMachineLearning (4)/computer_vision_course_materials/bird.jpg"

original_image = cv2.imread(COLOR_IMAGE_FILE, cv2.IMREAD_COLOR)

# We have to transform the image into grayscale in order to detect
# edges accurately using the edge detection kernel

# OpenCV handles BGR instead of RGB
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Laplacian kernel
edge_kernel = np.array(
    [
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ]
)

# `-1` defines the destination depth meaning that the depth
# will be the same as the original image
# This means the filter result will have the same dimension
# as the input image
edge_image = cv2.filter2D(gray_image, -1, edge_kernel)

# OpenCV also has their own Laplacian kernel built in
ocv_edge_image = cv2.Laplacian(gray_image, -1)

# Show the results
cv2.imshow("Original Image", original_image)
cv2.imshow("Edge Detection Image", edge_image)
cv2.imshow("OpenCV Edge Detection Image", ocv_edge_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
