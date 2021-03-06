import cv2
import dlib


# ----------------------------------------------------------------------------
# function used to resize image
# input: image, dim = (x,y)
# output: image
def resizeImage(img, dim=(650, 650)):
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


# ----------------------------------------------------------------------------
# function used to intilize dlib objects
# input: none
# output: objects
def initializeDlib():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("../dlib-predictor.dat")
    return detector, predictor


# ----------------------------------------------------------------------------
# function used to rotate image
# input: image
# output: image rotated
def rotateImage(img):
    rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return rotated


# ----------------------------------------------------------------------------
# function used to exctract frames from video
# input: video
# output: frames
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
