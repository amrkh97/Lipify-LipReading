from FR import *
from lip_detection import *


# ----------------------------------------------------------------------------
# function used to extract lips points
# input: frame Name + extension
# output: frame, mouth_roi points pair vector
def extractLips(fileName):
    img = readFrame(fileName)
    detector, predictor = initializeDlib()

    resized = resizeImage(img)

    inputFrame, mouthROI = lipDetection(resized, detector, predictor)
    inputFrame, mouthRegion = mouthRegionExtraction(inputFrame, mouthROI)
    return inputFrame, mouthRegion, mouthROI


def extractLipsFromFrame(inputFrame):
    """Function to extract lips from a single frame"""
    detector, predictor = initializeDlib()
    resized = resizeImage(inputFrame)
    inputFrame, mouthROI = lipDetection(resized, detector, predictor)
    inputFrame, mouthRegion = mouthRegionExtraction(inputFrame, mouthROI)
    return mouthRegion


# ----------------------------------------------------------------------------
# function used to extract lips region
# input: frame, mouth_roi points pair vector
# output: mouth region image
def mouthRegionExtraction(inputFrame, mouthRoi):
    x0 = mouthRoi[0][0]
    x0 = x0 - 10
    y0 = mouthRoi[2][1]
    y0 = y0 - 10
    x1 = mouthRoi[6][0]
    x1 = x1 + 10
    y1 = mouthRoi[9][1]
    y1 = y1 + 10
    mouthPart = inputFrame[y0: y1, x0: x1]
    return inputFrame, mouthPart


if "__main__" == __name__:
    filename = "../test.jpg"
    frame, mouth, mouth_roi = extractLips(filename)
cv2.imshow("d", mouth)
cv2.waitKey(0)
cv2.destroyAllWindows()