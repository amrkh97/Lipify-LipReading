import cv2

from GaborSupportFunctions import readFrame, resizeImage, smoothImg, binaryImage
from SkinModel import getSkin
from getFace import extractFaceSkin, get_face_box, draw_face_box, extractMouthROI


def getMouth(smoothed, img, skin):
    face, status = extractFaceSkin(smoothed, skin)
    if not status:
        return img, False
    # get binary cleaned mask
    binary_cleaned_skin = binaryImage(face)
    # draw box
    boundig_boxes, croped_img = get_face_box(binary_cleaned_skin, img)
    if len(boundig_boxes) == 0:
        return img, False
    face_image, y0, y1, x0, x1 = draw_face_box(img, boundig_boxes, croped_img)
    mouth_region = extractMouthROI(img, y0, y1, x0, x1)
    cv2.imshow("mouth_region", mouth_region)
    cv2.waitKey(0)


# -----------------------------------------------
img = readFrame()
# -----------------------------------------------
resized = resizeImage(img)
# -----------------------------------------------
smoothed = smoothImg(resized)
# -----------------------------------------------
skin = getSkin(smoothed)
cv2.imshow("skin", skin)
cv2.waitKey(0)
getMouth(smoothed, resized, skin)
# -----------------------------------------------
