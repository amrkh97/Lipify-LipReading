import time

from FR import *
from lip_detection import *


def getVideoFrames(videoPath):
    """Function to return a video's frames in a list
    :type videoPath: String
    """
    vidcap = cv2.VideoCapture(videoPath)
    success, image = vidcap.read()
    allFrames = []
    while success:
        allFrames.append(image)
        success, image = vidcap.read()
    return allFrames


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
    mouthRegion = cv2.resize(mouthRegion, (150, 100))

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

    startTime = time.time()
    videoPath = "../New-DataSet-Videos/S1/Adverb/again_0.mp4"
    frames = getVideoFrames(videoPath)
    detected = []
    for i, frame in enumerate(frames):
        detected.append(extractLipsFromFrame(frame))
        cv2.imshow(str(i), detected[-1])

    print("Run Time: {} Seconds".format(time.time() - startTime))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
