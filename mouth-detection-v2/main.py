import cv2
import numpy as np
import skimage.io as io
from FR import *
from SkinSegmentation import *
from FaceDetection import *
from mouthDetection import *

# frames reading
# function that calls (FR.py) functions that is used to inquire a filter cleared image of frame (preprossesing)
img = readFrame()
cv2.waitKey(0)
# resize image
resized = resizeImage(img)
# smooth image to remove noise
smoothed_img = smoothImg(resized)
# get skin mask
skin_mask = segmentSkin(smoothed_img)
# remove background
skin_img = extractSkin(smoothed_img,skin_mask)
# get binary cleaned mask
binary_cleaned_skin = binaryImage(skin_img)
# draw box
boundig_boxes, croped_img = get_box(binary_cleaned_skin,resized)
face_image,y0,y1,x0,x1 = draw_RGB_with_Rect(resized,boundig_boxes, croped_img)
mouth_region = extractMouthArea(resized,y0,y1,x0,x1)
mouth = mouthExtraction(mouth_region)
# drawMouthContour(mouth)
boundig_boxes2, croped_img2 = get_box2(mouth,mouth_region)
all_zeros = not np.count_nonzero(croped_img2)
if all_zeros != True:
    mouthFinal = draw_RGB_with_Rect2(mouth_region,boundig_boxes2, croped_img2)
    cv2.imshow("cont2", mouthFinal)
    cv2.waitKey(0)
else:
    cv2.imshow("cont2", mouth_region)
    cv2.waitKey(0)

