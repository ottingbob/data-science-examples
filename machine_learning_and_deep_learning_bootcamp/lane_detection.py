import cv2
import numpy as np


def draw_the_lines(image, lines):
    # Create the distinct image for the lines - all 0 values means black image
    lines_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # There are (x,y) coordinates for the starting and ending points of
    # the lines
    for line in lines:
        for x1, y1, x2, y2 in line:
            # OpenCV uses BGR so we will have blue lines show up here
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), thickness=3)

    # Finally merge the image with the lines
    # We have the transparency of each image as the following arg
    image_with_lines = cv2.addWeighted(image, 0.8, lines_image, 1, 0.0)
    return image_with_lines


def region_of_interest(image, region_points):
    # We are going to replace pixels with 0 (black) color
    # These regions we are not interested in
    #
    # Create the mask
    mask = np.zeros_like(image)

    # The region we are interested in is the lower triangle - 255 white pixels
    cv2.fillPoly(mask, region_points, 255)

    # We have to use the mask by keeping the regions of the original image
    # where the mask has white colored pixels
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def get_detected_lanes(frame_data):
    (height, width) = frame_data.shape[0], frame_data.shape[1]
    # print(height)
    # 360
    # print(width)
    # 640

    # We have to turn the image into grayscale for edge detection
    gray_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2GRAY)

    # Apply edge detection kernel - Canny's algorithm
    # `100` is the lower threshold
    # `120` is the higher threshold
    #
    # This is an `optimal detector` in that:
    # - It has low error rate - can detect existing edges only
    # - It has good localization - detected edges and real edges are more or
    #   less the same
    # - Every edge is only detected once
    #
    # Pixel gradients higher than the upper threshold are accepted as an edge
    # Gradient values below the lower threshold are rejected
    # Values in between will be accepted only if it connected to a pixel that
    #   that is above the upper threshold
    canny_frame = cv2.Canny(gray_frame, 100, 120)

    # There is no need to look at the whole image. We just want to consider the
    # region right in front of the car
    #
    # We will create a mask and then we have to apply a `logical AND` on the
    # pixels in the frame
    region_of_interest_vertices = [
        # Bottom left corner
        (0, height),
        # Just below middle of the image
        (width / 2, height * 0.65),
        # Bottom right corner
        (width, height),
    ]

    # We can get rid of unrelevant part of the image and just keep the lower
    # triangle region
    cropped_image = region_of_interest(
        canny_frame, np.array([region_of_interest_vertices], "int32")
    )

    # Use the line detection algorithm - Hough transformation
    lines = cv2.HoughLinesP(
        cropped_image,
        # `rho` defines 2 pixels which is the distance from the origin
        rho=2,
        # `theta` is divided by `180` since we are dealing with radians instead
        #   of degrees - 1 degree = pi / 180
        # We initialize this to 1 degree
        theta=np.pi / 180,
        # Minimum number of intersections to detect a given line
        # `50` means if more than 50 curves intersect we consider the edge as
        #   a valid line on the image
        threshold=50,
        lines=np.array([]),
        # Line segments shorter than this are rejected
        minLineLength=40,
        # Max allowed gap between points on the same line to link them
        maxLineGap=150,
    )

    # Draw the lines on the image
    image_with_lines = draw_the_lines(frame_data, lines)

    # return cropped_image
    return image_with_lines


LANE_DETECTION_VIDEO = "mldl_bootcamp_resources/PythonMachineLearning (4)/computer_vision_course_materials/lane_detection_video.mp4"

# The video will contain several frames
video = cv2.VideoCapture(LANE_DETECTION_VIDEO)

while video.isOpened():
    # Read the frames on a 1 by 1 basis
    is_grabbed, frame = video.read()
    # End of video or the frame was unable to be read
    if not is_grabbed:
        break

    frame = get_detected_lanes(frame)

    cv2.imshow("Lane Detection Video", frame)
    cv2.waitKey(20)

video.release()
cv2.destroyAllWindows()
