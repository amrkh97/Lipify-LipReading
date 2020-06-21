# ----------------------------------------------------------------------------
# function used to extract mouth ROI from image using face coords
# input: Single frame, Face coords (y, y+h, x, x+w)
# output: Mouth ROI
def extractMouthROI(img, y0, y1, x0, x1):
    N = y1 - y0
    M = x1 - x0
    x2 = x0 + int(M / 3)
    y2 = y0 + int((5 * N / 8))
    x3 = x2 + int((2 * M / 5))
    y3 = y2 + int((N / 3))
    mouth = img[y2: y3, x2: x3]
    return mouth
