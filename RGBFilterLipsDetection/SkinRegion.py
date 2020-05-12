import cv2
import numpy as np

def checkWhitePixelCount(skin_mask, regions):
    maxIndex = 0
    inRatio = 0
    for i, region in enumerate(regions):
        part = np.copy(skin_mask[region[0]:region[1], region[2]:region[3]])
        ret, bw_part = cv2. threshold(part,127,255,cv2. THRESH_BINARY)
        ratio = np.sum(bw_part == 255)/ np.sum(bw_part != 255)
        if ratio >= inRatio:
            inRatio = ratio
            maxIndex = i
    x1, x2, y1, y2 = regions[maxIndex]
    cv2.rectangle(skin_mask, (x1,y1), (x2,y2), (255, 0, 0), 2)
    cv2.imshow("face", skin_mask)
    cv2.waitKey(0)
    return

#---------------------------------------------------------
def extractSkinRegions(img, skinMask):
    BLUR = 21
    img2 = np.copy(img)
    CANNY_THRESH_1 = 10
    CANNY_THRESH_2 = 200
    MASK_DILATE_ITER = 10
    MASK_ERODE_ITER = 10
    MASK_COLOR = (0, 0, 0)
    edges = cv2.Canny(skinMask, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    contour_info = []
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return [], False
    regionCheck = []
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        if w > 80 and h > 80:
            x1 = x
            x2 = x+w
            y1 = y
            y2 = y+h
            regionCheck.append((x1,x2,y1,y2))
    checkWhitePixelCount(skinMask, regionCheck)
    return