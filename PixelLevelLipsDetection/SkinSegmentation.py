import cv2
import numpy as np

from FR import operation
from MatrixOperations import logic_and_list, logic_or_list, createMatrix, logic_not
from canny import cannyDetection, getGrayImage, gaussian_blur2
from colorspace import getHSV


# ----------------------------------------------------------------------------
# function used to exract skin from image
# input: image
# output: skin mask
def segmentSkin(img):
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]
    # ----------------------------------------------------
    h, s, v = getHSV(img)
    # ----------------------------------------------------
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = (B - Y) * 0.564 + 128
    Cr = (R - Y) * 0.713 + 128
    # ----------------------------------------------------
    skin_one = createMatrix(len(img), len(img[0]), 0)
    # ----------------------------------------------------
    skin_one_h = logic_and_list(h >= 0.0, h <= 50.0)
    # ----------------------------------------------------
    skin_one_s = logic_and_list(s >= 0.23, s <= 0.68)
    # ----------------------------------------------------
    logic1 = logic_and_list(R > 95, G > 40)
    logic2 = logic_and_list(logic1, B > 20)
    logic3 = logic_and_list(logic2, R > G)
    logic4 = logic_and_list(logic3, R > B)
    logic5 = np.absolute(np.array(R) - np.array(G))
    skin_one_rgb = logic_and_list(logic4, logic5 > 15)
    # ----------------------------------------------------
    skin_one = logic_and_list(logic_and_list(skin_one_h, skin_one_rgb), skin_one_s)
    # ----------------------------------------------------
    skin_two = createMatrix(len(img), len(img[0]), 0)
    # ----------------------------------------------------
    logic1 = logic_and_list(R > 95, G > 40)
    logic2 = logic_and_list(logic1, B > 20)
    logic3 = logic_and_list(logic2, R > G)
    logic4 = logic_and_list(logic3, R > B)
    logic5 = np.absolute(np.array(R) - np.array(G))
    skin_two_rgb = logic_and_list(logic4, logic5 > 15)
    # ----------------------------------------------------
    logic1 = logic_and_list(Cr > 135, Cb > 85)
    logic2 = logic_and_list(logic1, Y > 80)
    logic3 = logic_and_list(logic2, Cr <= ((1.5862 * Cb) + 20))
    logic4 = logic_and_list(logic3, Cr >= ((0.3448 * Cb) + 76.2069))
    logic5 = logic_and_list(logic4, Cr >= ((-4.5652 * Cb) + 234.5652))
    logic6 = logic_and_list(logic5, Cr <= ((-1.15 * Cb) + 301.75))
    skin_two_YCbCr = logic_and_list(logic6, Cr <= ((-2.2857 * Cb) + 432.85))
    # ----------------------------------------------------
    skin_two = logic_and_list(skin_two_YCbCr, skin_two_rgb)
    # ----------------------------------------------------
    skin = logic_or_list(skin_one, skin_two)
    # ----------------------------------------------------
    skin_image = np.copy(img)
    # ----------------------------------------------------
    skin_image[skin] = 255
    # ----------------------------------------------------
    skin_image[logic_not(skin)] = 0
    # ----------------------------------------------------
    holes_filled_skin_image = fill_holes(skin_image)
    holes_filled_skin_image = holes_filled_skin_image.astype(np.uint8)
    cv2.imshow('skin_image', holes_filled_skin_image)
    return holes_filled_skin_image


# ----------------------------------------------------------------------------
# function used to support segmentSkin function by filling holes in the skin mask from image
# input: skin mask
# output: skin mask
def fill_holes(img):
    img = getGrayImage(img)
    ret, thresh_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    res = operation(thresh_img, kernel, 1, "erosion")
    res = operation(res, kernel, 1, "dilation")
    return res


# ----------------------------------------------------------------------------
# function used to exract skin from image
# input: image, skin mask
# output: skin only image
def extractSkin(img, skinMask):
    BLUR = 21
    img2 = np.copy(img)
    CANNY_THRESH_1 = 10
    CANNY_THRESH_2 = 200
    MASK_DILATE_ITER = 10
    MASK_ERODE_ITER = 10
    MASK_COLOR = (0, 0, 0)
    edges = cannyDetection(skinMask, CANNY_THRESH_1, CANNY_THRESH_2)
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])
    edges = operation(edges, kernel, 0, "dilation")
    edges = operation(edges, kernel, 0, "erosion")
    contour_info = []
    edges = edges.astype("uint8")
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cntf = []
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    cv2.drawContours(img, contours, -1, 255, 3)
    cv2.imshow('contours', img)
    cv2.waitKey(0)
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    # --------------------------
    max_contour = contour_info[0]

    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))
    mask = operation(mask, kernel, 0, "dilation")
    mask = operation(mask, kernel, 0, "erosion")

    mask = gaussian_blur2(mask, BLUR, 0)

    mask_stack = np.dstack([mask] * 3)

    mask_stack = mask_stack.astype('float32') / 255.0
    img = img.astype('float32') / 255.0

    masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)
    masked = (masked * 255).astype('uint8')
    return masked
