import cv2

MULTI_FACE_IMAGE = "mldl_bootcamp_resources/machine_learning (actual one)/people.jpg"
# OpenCV has a lot of pre-trained classifiers for face detection,
# eye detection, etc.
BOOSTING_MODEL = "mldl_bootcamp_resources/machine_learning (actual one)/haarcascade_frontalface_alt.xml"

# This loads the XML file with the trained boosting model
cascade_classifier = cv2.CascadeClassifier(BOOSTING_MODEL)

multi_face_image = cv2.imread(MULTI_FACE_IMAGE)

# Convert the image into grayscale
# OpenCV handles BGR instead of RGB
gray_image = cv2.cvtColor(multi_face_image, cv2.COLOR_BGR2GRAY)

# This will return the rectangles that the classifier identifies there
# to be faces in
detected_faces = cascade_classifier.detectMultiScale(
    # We pass in the grayscale image to detect the faces
    gray_image,
    # Scale factor compensates for faces being closer or further to the camera
    # and specifies how much the image is reduced at each image scale
    # The model has a fixed size during training so by rescaling the input image
    # you can resize a larger face to a smaller one, making it detectable by
    # the algorithm
    # Values: 1.1 - 1.4
    #   Smaller -> algorithm will be slow since it is more thorough
    #   Higher  -> faster detection with the risk of missing some faces
    #              altogether
    scaleFactor=1.1,
    # Specifies how many neighbors each candidate rectangle should have to
    # retain it
    # Higher values -> less detections but with higher quality
    # Lower values  -> introduces more false positives
    minNeighbors=2,
    # Ignores objects smaller than the given size. We can specify what is the
    # smallest object that we want to consider / recognize
    # [30x30] is the standard size
    minSize=(30, 30),
)

for (x, y, width, height) in detected_faces:
    cv2.rectangle(
        # Image to draw on
        multi_face_image,
        # Starting point -- top left corner of rectangle
        (x, y),
        # Ending point -- bottom right corner of rectangle
        (x + width, y + height),
        # Color to draw - OpenCV uses BGR so this means it will be red
        (0, 0, 255),
        # Thickness of the color
        4,
    )


cv2.imshow("Mutli Face Image", multi_face_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
