import cv2
import numpy as np


def extractFaceSkin(img, skinMask):
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
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return [], False
    cntf = []
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    cv2.drawContours(img, contours, -1, 255, 3)
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    # --------------------------
    max_contour = contour_info[0]
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)

    mask_stack = np.dstack([mask] * 3)

    mask_stack = mask_stack.astype('float32') / 255.0
    img = img.astype('float32') / 255.0

    masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)
    masked = (masked * 255).astype('uint8')
    return masked, True
# ----------------------------------------------------------------------------
# function used to get box of the face from image
# input: skin image, frame
# output: bounding box of image
def get_face_box(img, RGB_img):
    bounding_boxes = []
    masks = np.zeros(img.shape).astype("uint8")
    output = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)
    index = 0
    num_labels = output[0]
    labels = output[1]
    stats = output[2]

    if num_labels > 0:
        box_ratio_upper_bound = 1.1
        box_ratio_lower_bound = 0.4
        area = 0
        index = 0
        for i in range(0, labels.max() + 1):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            if x + w < y + h:
                minor_axis = x + w
                major_axis = y + h
            else:
                minor_axis = y + h
                major_axis = x + w
            if ((w / h < box_ratio_upper_bound) and (w / h > box_ratio_lower_bound) and (
                    minor_axis / major_axis > 0.25) and (minor_axis / major_axis < 0.97) and area > 1500):
                if w / h <= 0.7:
                    h = int(h - 0.3 * h)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                mask = np.zeros(img.shape).astype("uint8")
                mask[y:y + h, x:x + w] = 255
                bounding_boxes.append([y, y + h, x, x + w])
                if not ((mask == 255).all()):
                    masks += mask
                index += 1
        masks //= 255
        masks[masks == 1] = 255
        masks[masks > 0] = 255
        img[masks == 0] = 0
        return bounding_boxes, img


# ----------------------------------------------------------------------------
# function used to draw box on face in the frame
# input: frame, bounding box data, cut region
# output: frame with box drown, coord of box
def draw_face_box(RGB_image, Boundary_boxes, cp):
    ret, thresh = cv2.threshold(cp, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, width, height = cv2.boundingRect(contours[0])
    y0 = y
    y1 = y + height
    x0 = x
    x1 = x + width
    return RGB_image, y0, y1, x0, x1
# ----------------------------------------------------------------------------
def extractMouthROI(img, y0, y1, x0, x1):
    N = y1 - y0
    M = x1 - x0
    x2 = x0 + int(M / 3)
    y2 = y0 + int((5 * N / 8))
    x3 = x2 + int((2 * M / 5))
    y3 = y2 + int((N / 2))
    mouth = img[y2: y3, x2: x3]
    return mouth