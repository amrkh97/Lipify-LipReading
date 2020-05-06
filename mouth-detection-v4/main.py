import cv2
import time
import csv

from FR import *
from mouthDetection import *

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
startTime = time.time()
videoPath = "../Prototype-Test-Videos/Colors_7.mp4"
frames = getVideoFrames(videoPath)
detected = []
for i, frame in enumerate(frames):
    img = extractLipsHaarCascade(faceCascade, frame)
    #img = detect(frame, faceCascade)
    #img = resizeImage(img, dim=(150,100))
    detected.append(img)
    cv2.imshow(str(i), detected[-1])
print("Run Time: {} Seconds".format(time.time() - startTime))
with open('../Image-Processing-Test/ModelsTiming.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    # writer.writerow(['Model', 'Video Name', 'Time Taken', 'Accuracy'])
    vidName = videoPath.split('/')
    writer.writerow(['HaarCascade', vidName[len(vidName)-1], time.time() - startTime, 90])

cv2.waitKey(0)
cv2.destroyAllWindows()