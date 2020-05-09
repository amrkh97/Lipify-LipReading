import cv2
import time
import csv

from FR import *
from mouthDetection import *

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# Adverb_1
for i in range(1,1000):
    startTime = time.time()
    videoPath = "../Prototype-Test-Videos/s2/s2 "+"("+str(i)+")"+".mpg"
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
        #cv2.imshow(str(i), detected[-1])

    accuracy = (corrcount/len(frames))*100
    print("Run Time: {} Seconds".format(time.time() - startTime))
    with open('../Project_Insights/ModelsTiming.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        # writer.writerow(['Model', 'Video Name', 'Time Taken', 'Accuracy'])
        vidName = videoPath.split('/')
        writer.writerow(['HaarCascade', vidName[len(vidName)-1], time.time() - startTime, accuracy])

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()