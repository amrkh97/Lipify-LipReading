import cv2

from GaborSupportFunctions import readFrame, resizeImage, smoothImg, binaryImage
from SkinModel import getSkin
from SkinRegion import extractSkinRegions


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
extractSkinRegions(smoothed, skin)
# -----------------------------------------------
