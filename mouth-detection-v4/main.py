import cv2
import numpy as np
from FR import readFrame
from mouthDetection import detect

img = readFrame()
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img = detect(img, faceCascade)
cv2.imshow("face Detected", img)
cv2.waitKey(0)