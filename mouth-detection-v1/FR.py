import cv2
import numpy as np


# ----------------------------------------------------------------------------
# function used to smooth from image
# input: image
# output: image
def smoothImg(img):
    blur = cv2.bilateralFilter(img, 9, 75, 75)
    cv2.imshow('blur', blur)
    return blur


# ----------------------------------------------------------------------------
# function used to sharp edges from image
# input: image
# output: image
def sharpenEdges(img):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpend = cv2.filter2D(img, -1, kernel)
    cv2.imshow('sharped', sharpend)
    return sharpend


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
# function used to errode image
# input: image
# output: image
def erosion_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data), len(data[0])))
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


# ----------------------------------------------------------------------------------------
# function used to dialte image 
# input: image
# output: image
def dilation_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data), len(data[0])))
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
