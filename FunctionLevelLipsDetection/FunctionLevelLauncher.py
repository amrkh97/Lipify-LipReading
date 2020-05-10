from FaceBoundry import *
from SkinExtraction import *

from SuportFunctions import *
from LipsBoundry import *
#-----------------------------------------------------------------
def getMouth(img):
    resized = resizeImage(img)
    # smooth image to remove noise
    smoothed_img = smoothImg(resized)
    # get skin mask
    skin_mask = segmentSkin(smoothed_img)
    # remove background
    skin_img, status = extractSkin(smoothed_img, skin_mask)
    if status == False:
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
    boundig_boxes2, croped_img2 = get_box2(mouth, mouth_region)
    all_zeros = not np.count_nonzero(croped_img2)
    if all_zeros != True:
        mouthFinal = draw_RGB_with_Rect2(mouth_region, boundig_boxes2, croped_img2)
        mouthFinal = resizeImage(mouthFinal, (150, 100))
        return mouthFinal, True
    else:
        mouth_region = resizeImage(mouth_region, (150, 100))
        return mouth_region, True
    return img, False
#------------------------------------------------------------------------
if "__main__" == __name__:
    videoPath = "../Prototype-Test-Videos/Adverb_1.mp4"
    frames, status = getVideoFrames(videoPath)
    if status == True:
        detected = []
        for i, frame in enumerate(frames):
            lips, status = getMouth(frame)
            if status == False:
                print("failed to get face")
                frame = resizeImage(frame, (150, 100))
                detected.append(frame)
            else:
                detected.append(lips)
            cv2.imshow(str(i), detected[-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed To Get Video")

