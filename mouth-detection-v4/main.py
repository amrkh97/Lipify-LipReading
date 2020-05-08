import cv2
import time
import csv

from FR import *
from mouthDetection import *

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
startTime = time.time()
# Adverb_1
videoPath = "../Prototype-Test-Videos/Adverb_1.mp4"
frames = getVideoFrames(videoPath)
detected = []
corrcount = 0
for i, frame in enumerate(frames):
    img, status = extractLipsHaarCascade(faceCascade, frame)
    if status == True:
        corrcount += 1
    #img = detect(frame, faceCascade)
    #img = resizeImage(img, dim=(150,100))
    detected.append(img)
    # cv2.imshow(str(i), detected[-1])

accuracy = (corrcount/len(frames))*100
print("Run Time: {} Seconds".format(time.time() - startTime))
with open('../Image-Processing-Test/ModelsTiming.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    # writer.writerow(['Model', 'Video Name', 'Time Taken', 'Accuracy'])
    vidName = videoPath.split('/')
    writer.writerow(['HaarCascade', vidName[len(vidName)-1], time.time() - startTime, accuracy])

cv2.waitKey(0)
cv2.destroyAllWindows()