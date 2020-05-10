import cv2


def extractMouthArea(img, y0, y1, x0, x1):
    N = y1 - y0
    M = x1 - x0
    x2 = x0 + int(M / 6)
    y2 = y0 + int((5 * N / 8))
    x3 = x2 + int((2 * M / 3))
    y3 = y2 + int((N / 2))
    cv2.rectangle(img, (x2, y2), (x3, y3), (0, 255, 0), 2)
    return img
