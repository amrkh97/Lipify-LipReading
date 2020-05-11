import cv2
import numpy as np


# ----------------------------------------------------------------------------
# function used to read Video and get its frames
# input: Video Path
# output: List of frames
def getVideoFrames(videoPath):
    """Function to return a video's frames in a list
    :type videoPath: String
    """
    vidcap = cv2.VideoCapture(videoPath)
    if not vidcap.isOpened():
        return [], False
    success, image = vidcap.read()
    allFrames = []
    while success:
        allFrames.append(image)
        success, image = vidcap.read()
    return allFrames, True


# ----------------------------------------------------------------------------
# function used to smooth from image
# input: image
# output: image
def smoothImg(img):
    blur = cv2.bilateralFilter(img, 9, 75, 75)
    return blur


# ----------------------------------------------------------------------------
# function used to resize image
# input: image, dim = (x,y)
# output: image
def resizeImage(img, dim=(650, 650)):
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

# ----------------------------------------------------------------------------
# function used to change image to binary
# input: image
# output: binary image
def binaryImage(img):
    binary_image = np.copy(img)
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    binary_image[binary_image > 0] = 255
    return binary_image


# ----------------------------------------------------------------------------
# function used to change image to binary
# input: image
# output: binary image
def binaryImage2(img):
    binary_image = np.copy(img)
    # binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    binary_image[binary_image > 0] = 255
    binary_image[binary_image < 0] = 0
    binary_image = binary_image.astype(np.uint8)
    binary_image = np.invert(binary_image)
    return binary_image
