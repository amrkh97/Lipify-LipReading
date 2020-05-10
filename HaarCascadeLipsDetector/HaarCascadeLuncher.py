import cv2
from FramesManipulations import *
from HaarDetector import extractLipsHaarCascade

if "__main__" == __name__:
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    videoPath = "../Prototype-Test-Videos/Colors_2.mp4"
    frames, status = getVideoFrames(videoPath)
    if status == True:
        detected = []
        for i, frame in enumerate(frames):
            img, status = extractLipsHaarCascade(faceCascade, frame)
            detected.append(img)
            cv2.imshow(str(i), detected[-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("error in getting video")