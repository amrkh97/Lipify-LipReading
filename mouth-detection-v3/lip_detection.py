import cv2
import numpy as np
import dlib


def lipDetection(frame,detector,predictor):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return 0, 0
    else:
        face = faces[0]
        landmarks = predictor(gray, face)
        mouth_roi = []
        for n in range(48,68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            mouth_roi.append((x,y))
            cv2.circle(frame,(x,y),3,(255,0,0),1)
        return frame, mouth_roi