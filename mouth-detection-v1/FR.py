import cv2
import numpy as np
import skimage.io as io
#----------------------------------------------------------------------------
# function used to read a frame from file
# input: none
# output: image
def readFrame():
    img = cv2.imread('DataSet-Trial/close.png')
    return img
#----------------------------------------------------------------------------
# function used to smooth from image
# input: image
# output: image
def smoothImg(img):
    blur = cv2.bilateralFilter(img,9,75,75)
    cv2.imshow('blur', blur)
    return blur
#----------------------------------------------------------------------------
# function used to sharp edges from image
# input: image
# output: image
def sharpenEdges(img):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpend = cv2.filter2D(img, -1, kernel)
    cv2.imshow('sharped', sharpend)
    return sharpend
#----------------------------------------------------------------------------
# function used to resize image
# input: image, dim = (x,y)
# output: image
def resizeImage(img, dim=(650,650)):
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized
#----------------------------------------------------------------------------
# function used to change image to binary
# input: image
# output: binary image
def binaryImage(img):
    binary_image = np.copy(img)
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    binary_image[binary_image>0] = 255
    return binary_image
#----------------------------------------------------------------------------
# function used to errode image
# input: image
# output: image
def erosion_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[0]
            temp = []
    return data_final
#----------------------------------------------------------------------------------------
# function used to dialte image 
# input: image
# output: image
def dilation_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[-1]
            temp = []
    return data_final