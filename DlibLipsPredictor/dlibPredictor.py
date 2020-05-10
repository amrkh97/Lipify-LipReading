import cv2


def lipDetection(frame, detector, predictor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return frame, [], [], False
    else:
        face = faces[0]
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        faceCoords = [x,y,w,h]
        landmarks = predictor(gray, face)
        mouth_roi = []
        for n in range(48, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            mouth_roi.append((x, y))
        return frame, mouth_roi, faceCoords, True
