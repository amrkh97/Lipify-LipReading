import cv2
import numpy as np

from dlibPredictor import lipDetection
from dlibSupportFunctions import resizeImage, initializeDlib, getVideoFrames


# ----------------------------------------------------------------------------
# function used to extract lips points
# input: frame, detector, predictor
# output: Mouth Region, Status
def extractLipsFromFrame(inputFrame, detector, predictor):
    """Function to extract lips from a single frame"""
    img = np.copy(inputFrame)
    resized = resizeImage(inputFrame)
    inputFrame, mouthROI, faceCoords, status = lipDetection(resized, detector, predictor)
    if not status:
        return inputFrame, False
    inputFrame, mouthRegion = mouthRegionExtraction(inputFrame, mouthROI, faceCoords)
    mouthRegion = cv2.resize(mouthRegion, (150, 100))
    mouthRegion = cv2.cvtColor(mouthRegion, cv2.COLOR_BGR2GRAY)
    return mouthRegion, True


# ----------------------------------------------------------------------------
# function used to extract lips region
# input: frame, mouth_roi points pair vector
# output: mouth region image
def mouthRegionExtraction(inputFrame, mouthRoi, faceCoords):
    x0 = faceCoords[0]
    y0 = mouthRoi[2][1]
    y0 = y0 - 20
    x1 = faceCoords[0] + faceCoords[2]
    y1 = faceCoords[1] + faceCoords[3] + 20
    mouthPart = inputFrame[y0: y1, x0: x1]
    mouthPart = cv2.resize(mouthPart, (150, 100))
    return inputFrame, mouthPart

# ----------------------------------------------------------------------------
# Main
if "__main__" == __name__:
    videoPath = "../Prototype-Test-Videos/Adverb_1.mp4"
    frames, status = getVideoFrames(videoPath)
    if status:
        detector, predictor = initializeDlib()
        detected = []
        for i, frame in enumerate(frames):
            lips, status = extractLipsFromFrame(frame, detector, predictor)
            if not status:
                print("failed to get face")
                detected.append(frame)
            else:
                detected.append(lips)
            cv2.imshow(str(i), detected[-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Video Was Not Found")
