import cv2
import numpy as np
import skimage.io as io
import math

def extractMouthArea(img,y0,y1,x0,x1):
    # part = int((img.shape[0]*2)/3)
    # mouth = img[part : img.shape[0] , : ]
    N = y1-y0
    M = x1-x0
    x2=x0 + int(M/6)
    y2=y0 + int((5*N/8))
    # y2=y0 + int((5*N/6))
    x3= x2+ int((2*M/3))
    y3= y2+int((N/2))
    # y3= y2+int((N/3))
    cv2.rectangle(img,(x2,y2),(x3,y3),(0,255,0),2)
    # mouth = img[y2 : y3 , x2 : x3 ]
    return img