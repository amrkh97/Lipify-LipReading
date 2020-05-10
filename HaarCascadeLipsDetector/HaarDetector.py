import cv2

def extractLipsHaarCascade(haarDetector, frame):
    """Function to extract lips from a frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray = 0
    status = False
    faces = haarDetector.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        roi_gray = cv2.resize(gray, (150, 100))
        return roi_gray, status
    for (x, y, w, h) in faces:
        roi_gray = gray[y + (2 * h // 3):y + h, x:x + w]
    roi_gray = cv2.resize(roi_gray, (150, 100))
    status = True
    return roi_gray, status