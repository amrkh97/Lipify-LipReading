import cv2

from FR import resizeImage, readFrame, smoothImg, binaryImage
from FaceDetection import get_box, draw_RGB_with_Rect
from SkinSegmentation import segmentSkin, extractSkin
from mouthDetection import extractMouthArea

# frames reading
# function that calls (FR.py) functions that is used to inquire a filter cleared image of frame (preprossesing)
img = readFrame()
# resize image
resized = resizeImage(img)
cv2.waitKey(0)
# smooth image to remove noise
smoothed_img = smoothImg(resized)
cv2.waitKey(0)
# get skin mask
skin_mask = segmentSkin(smoothed_img)
cv2.waitKey(0)
# remove background
skin_img = extractSkin(smoothed_img, skin_mask)
cv2.imshow('skin_img1', skin_img)
cv2.waitKey(0)
# get binary cleaned mask
binary_cleaned_skin = binaryImage(skin_img)
cv2.imshow('Binary_Image', binary_cleaned_skin)
cv2.waitKey(0)
# draw box
boundig_boxes, croped_img = get_box(binary_cleaned_skin, resized)
face_image, y0, y1, x0, x1 = draw_RGB_with_Rect(resized, boundig_boxes, croped_img)
cv2.waitKey(0)
cv2.imshow('skin_img', face_image)
cv2.waitKey(0)
mouth_region = extractMouthArea(resized, y0, y1, x0, x1)
cv2.imshow('mouth', mouth_region)
cv2.waitKey(0)
