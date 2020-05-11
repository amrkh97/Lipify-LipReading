import cv2
import numpy as np
from GaborSupportFunctions import *
from SkinModel import getSkin
#-----------------------------------------------
img = readFrame()
#-----------------------------------------------
resized = resizeImage(img)
#-----------------------------------------------
smoothed = smoothImg(resized)
#-----------------------------------------------
skin = getSkin(smoothed)
cv2.imshow("skin", skin*255)
cv2.waitKey(0)