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
   
    return frame, mouth_roi