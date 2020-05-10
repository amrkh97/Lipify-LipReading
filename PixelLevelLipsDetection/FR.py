import cv2
import numpy as np
from canny import getGrayImage
import math
# ----------------------------------------------------------------------------
# function used to read a frame from file
# input: none
# output: image
def readFrame():
    img = cv2.imread('DataSet-Trial/close.png')
    cv2.imshow('img', img)
    return img
# ----------------------------------------------------------------------------
# function used to smooth from image
# input: image
# output: image
def smoothImg(img):
    blur = cv2.bilateralFilter(img, 10, 75, 75)
    cv2.imshow('blur', blur)
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
# -----------------------------------------------------------------------------------------
# function that add padding to an image
# input: image, padding, kernal, operation type
# output: image
def add_padding(image, padding, value):
    return cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=value)
# -----------------------------------------------------------------------------------------
# function that does both erosion and dialition on image
# input: image, padding, kernal, operation type
# output: image
def operation(image, kernel, padding=0, operation=None):
    if operation:
        img_operated = image.copy()
        padding_value = 0
        if operation == "erosion":
            padding_value = 1
        padded = add_padding(image, padding, padding_value)
        vertical_window = padded.shape[0] - kernel.shape[0]  # final vertical window position
        horizontal_window = padded.shape[1] - kernel.shape[1]  # final horizontal window position
        vertical_pos = 0
        while vertical_pos <= vertical_window:
            horizontal_pos = 0
            while horizontal_pos <= horizontal_window:
                dilation_flag = False
                erosion_flag = False
                for i in range(kernel.shape[0]):
                    for j in range(kernel.shape[1]):
                        if kernel[i][j] == 1:
                            if operation == "erosion":
                                if padded[vertical_pos + i][horizontal_pos + j] == 0:
                                    erosion_flag = True
                                    break
                            elif operation == "dilation":
                                if padded[vertical_pos + i][horizontal_pos + j] == 1:
                                    dilation_flag = True
                                    break
                            else:
                                return "No Operation Chosen"
                    if operation == "erosion" and erosion_flag:
                        img_operated[vertical_pos, horizontal_pos] = 0
                        break
                    if operation == "dilation" and dilation_flag:
                        img_operated[vertical_pos, horizontal_pos] = 1
                        break
                horizontal_pos += 1
            vertical_pos += 1
        return img_operated
    return "Operation Required"
# -----------------------------------------------------------------------------------------
# function that extract frames from video
# input: video path
# output: List Of Images
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
