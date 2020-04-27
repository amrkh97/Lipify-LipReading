import cv2

from FR import readFrame
from FR import resizeImage
from mouthDetection import detect

img = readFrame()
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img = detect(img, faceCascade)
img = resizeImage(img, dim=(150,100))
cv2.imshow("face Detected", img)
cv2.waitKey(0)
