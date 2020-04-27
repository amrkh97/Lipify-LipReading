import cv2


def draw_boundry(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        #cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        #cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords


def detect(img, faceCascade):
    color = {"blue": (255, 0, 0)}
    coords = draw_boundry(img, faceCascade, 1.3, 5, color["blue"], "face")
    if len(coords) == 4:
        img = extractMouthArea(img, coords[1], coords[1] + coords[3], coords[0], coords[0] + coords[2])
    return img

def extractLipsHaarCascade(haarDetector, frame):
    """Function to extract lips from a frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray = 0
    faces = haarDetector.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        roi_gray = cv2.resize(gray, (150, 100))
        return roi_gray

    for (x, y, w, h) in faces:
        roi_gray = gray[y + (2 * h // 3):y + h, x:x + w]
    roi_gray = cv2.resize(roi_gray, (150, 100))
    return roi_gray

def extractMouthArea(img, y0, y1, x0, x1):
    # part = int((img.shape[0]*2)/3)
    # mouth = img[part : img.shape[0] , : ]
    N = y1 - y0
    M = x1 - x0
    x2 = x0 + int(2 * M / 6)
    y2 = y0 + int((5 * N / 7))
    # y2=y0 + int((5*N/6))
    x3 = x2 + int((2 * M / 5))
    y3 = y2 + int((N / 5))
    # y3= y2+int((N/3))
    cv2.rectangle(img, (x2, y2), (x3, y3), (0, 255, 0), 2)
    mouth = img[y2: y3, x2: x3]
    return mouth
