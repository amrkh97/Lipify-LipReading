import cv2
import numpy as np


def checkWhitePixelCount(skin_mask, regions):
    maxIndex = 0
    inRatio = 0
    if len(regions) == 0:
        return 0, 0, 0, 0, False
    for i, region in enumerate(regions):
        part = np.copy(skin_mask[region[0]:region[1], region[2]:region[3]])
        ret, bw_part = cv2.threshold(part, 127, 255, cv2.THRESH_BINARY)
        ratio = np.sum(bw_part == 255)
        if ratio >= inRatio:
            inRatio = ratio
            maxIndex = i
    x1, x2, y1, y2 = regions[maxIndex]
    idx, status = checkHolesInPart(skin_mask, regions, maxIndex)
    if status:
        y2 = int(y2 - 0.1 * y2)
        return x1, x2, y1, y2, True
    else:
        regions.pop(maxIndex)
        checkWhitePixelCount(skin_mask, regions)
    return 0, 0, 0, 0, False


def checkHolesInPart(skin_mask, regions, index):
    x1, x2, y1, y2 = regions[index]
    part = np.copy(skin_mask[y1:y2, x1:x2])
    contours, hierarchy = cv2.findContours(part, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    max_num = np.amax(hierarchy) + 1
    num_interior_contours = 0
    for c, h in zip(contours, hierarchy[0]):
        # If there is at least one interior contour, find out how many there are
        if h[2] != -1:
            # Make sure it's not the 'zero' contour
            if h[0] == -1:
                num_interior_contours = max_num - h[2]
            else:
                num_interior_contours = h[0] - h[2]
        else:
            num_interior_contours = 0
    if num_interior_contours >= 3:
        return True, index
    else:
        return False, index


# ---------------------------------------------------------
def extractSkinRegions(img, skinMask):
    BLUR = 21
    CANNY_THRESH_1 = 10
    CANNY_THRESH_2 = 200
    MASK_DILATE_ITER = 10
    MASK_ERODE_ITER = 10
    MASK_COLOR = (0, 0, 0)
    edges = cv2.Canny(skinMask, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return [], False
    regionCheck = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w > 80 and h > 80:
            x1 = x
            x2 = x + w
            y1 = y
            y2 = y + h
            regionCheck.append((x1, x2, y1, y2))
    if len(regionCheck) == 0:
        return [], False
    x1, x2, y1, y2, status = checkWhitePixelCount(skinMask, regionCheck)
    if status:
        return [y1, y2, x1, x2], True
    else:
        return [], False
