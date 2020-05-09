from FaceDetection import *
from SkinSegmentation import *

from FR import *
from mouthDetection import *
import time
import csv

def getMouth(img):
    # img = readFrame()
    # cv2.waitKey(0)
    # resize image
    resized = resizeImage(img)
    # smooth image to remove noise
    smoothed_img = smoothImg(resized)
    # get skin mask
    skin_mask = segmentSkin(smoothed_img)
    # remove background
    skin_img = extractSkin(smoothed_img, skin_mask)
    if len(skin_img) == 0:
        return img, False
    # get binary cleaned mask
    binary_cleaned_skin = binaryImage(skin_img)
    # draw box
    boundig_boxes, croped_img = get_box(binary_cleaned_skin, resized)
    if len(boundig_boxes) == 0:
        return img, False
    face_image, y0, y1, x0, x1 = draw_RGB_with_Rect(resized, boundig_boxes, croped_img)
    mouth_region = extractMouthArea(resized, y0, y1, x0, x1)
    mouth = mouthExtraction(mouth_region)
    # drawMouthContour(mouth)
    boundig_boxes2, croped_img2 = get_box2(mouth, mouth_region)
    all_zeros = not np.count_nonzero(croped_img2)
    if all_zeros != True:
        mouthFinal = draw_RGB_with_Rect2(mouth_region, boundig_boxes2, croped_img2)
        mouthFinal = resizeImage(mouthFinal, (150, 100))
        # cv2.imshow("cont2", mouthFinal)
        # cv2.waitKey(0)
        return mouthFinal, True
    else:
        mouth_region = resizeImage(mouth_region, (150, 100))
        # cv2.imshow("cont2", mouth_region)
        # cv2.waitKey(0)
        return mouth_region, True
    return img, False

# frames reading
# function that calls (FR.py) functions that is used to inquire a filter cleared image of frame (preprossesing)
startTime = time.time()
# Adverb_1
videoPath = "../Prototype-Test-Videos/Adverb_1.mp4"
frames = getVideoFrames(videoPath)
detected = []
# corrcount = 0
for i, frame in enumerate(frames):
    lips, status = getMouth(frame)
    if status == False:
        print("failed to get face")
        frame = resizeImage(frame, (150, 100))
        detected.append(frame)
    else:
        detected.append(lips)
        # corrcount+=1
    cv2.imshow(str(i), detected[-1])
    #cv2.imshow(str(i), inputframe)
# accuracy = (corrcount/len(frames))*100
# with open('../Image-Processing-Test/ModelsTiming.csv', 'a', newline='') as f:
#     writer = csv.writer(f)
#     # writer.writerow(['Model', 'Video Name', 'Time Taken', 'Accuracy'])
#     vidName = videoPath.split('/')
#     writer.writerow(['SkinDetectionAFunctions', vidName[len(vidName)-1], time.time() - startTime, accuracy])

print("Run Time: {} Seconds".format(time.time() - startTime))
cv2.waitKey(0)
cv2.destroyAllWindows()

