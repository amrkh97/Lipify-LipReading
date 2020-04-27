import cv2

from FR import *
from mouthDetection import *

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
videoPath = "../Prototype-Test-Videos/Colors_1.mp4"
frames = getVideoFrames(videoPath)
detected = []
for i, frame in enumerate(frames):
    img = extractLipsHaarCascade(faceCascade, frame)
    #img = detect(frame, faceCascade)
    #img = resizeImage(img, dim=(150,100))
    detected.append(img)
    cv2.imshow(str(i), detected[-1])
cv2.waitKey(0)
cv2.destroyAllWindows()