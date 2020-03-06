import cv2
import numpy as np
from FR import *
from lip_detection import *

#----------------------------------------------------------------------------
# function used to extract lips points
# input: frame Name + extension
# output: frame, mouth_roi points pair vector
def extractLips(filename):
    img = readFrame(filename)

    detector,predictor = initlizeDlib()

    resized = resizeImage(img)

    frame, mouth_roi = lipDetection(resized, detector, predictor)
    frame, mouth = mouthRegionExtraction(frame, mouth_roi)
    return frame, mouth, mouth_roi
#----------------------------------------------------------------------------
# function used to extract lips region
# input: frame, mouth_roi points pair vector
# output: mouth region image
def mouthRegionExtraction(frame, mouth_roi):
    x0 = mouth_roi[0][0]
    x0 = x0 - 10
    y0 = mouth_roi[2][1]
    y0 = y0 - 10
    x1 = mouth_roi[6][0]
    x1 = x1 + 10
    y1 = mouth_roi[9][1]
    y1 = y1 + 10
    mouth = frame[ y0 : y1 , x0 : x1 ]
    return frame , mouth