import cv2
import numpy as np
import dlib
#----------------------------------------------------------------------------
# function used to read a frame from file
# input: none
# output: image
def readFrame(filesname):
    img = cv2.imread('DataSet-Trial/'+filesname)
    return img
#----------------------------------------------------------------------------
# function used to resize image
# input: image, dim = (x,y)
# output: image
def resizeImage(img, dim=(650,650)):
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized
#----------------------------------------------------------------------------
# function used to intilize dlib objects
# input: none
# output: objects
def initlizeDlib():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    return detector,predictor


