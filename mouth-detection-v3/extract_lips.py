import time

import numpy as np

from FR import initializeDlib, resizeImage
from lip_detection import lipDetection


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


def extractLipsFromFrame(inputFrame):
    """Function to extract lips from a single frame"""
    img = np.copy(inputFrame)
    detector, predictor = initializeDlib()
    resized = resizeImage(inputFrame)
    inputFrame, mouthROI, faceCoords = lipDetection(resized, detector, predictor)
    if len(mouthROI) == 0:
        return []
    inputFrame, mouthRegion = mouthRegionExtraction(inputFrame, mouthROI, faceCoords)
    mouthRegion = cv2.resize(mouthRegion, (150, 100))
    mouthRegion = cv2.cvtColor(mouthRegion, cv2.COLOR_BGR2GRAY)
    return mouthRegion


# ----------------------------------------------------------------------------
# function used to extract lips region
# input: frame, mouth_roi points pair vector
# output: mouth region image
def mouthRegionExtraction(inputFrame, mouthRoi, faceCoords):
    # faceCoords = [x,y,w,h]
    # x0 = mouthRoi[0][0]
    # x0 = x0 - 20
    x0 = faceCoords[0]
    y0 = mouthRoi[2][1]
    y0 = y0 - 20
    x1 = faceCoords[0] + faceCoords[2]
    # x1 = mouthRoi[6][0]
    # x1 = x1 + 20
    # y1 = mouthRoi[9][1]
    # y1 = y1 + 20
    y1 = faceCoords[1] + faceCoords[3] + 20
    mouthPart = inputFrame[y0: y1, x0: x1]
    mouthPart = cv2.resize(mouthPart, (150, 100))
    return inputFrame, mouthPart


if "__main__" == __name__:

    startTime = time.time()
    # Adverb_1
    videoPath = "../Prototype-Test-Videos/Adverb_1.mp4"
    frames = getVideoFrames(videoPath)
    # for i in range(0,len(frames)):
    #     frames[i] = rotateImage(frames[i])
    #     #frames[i] = resizeImage(frames[i])   
    detected = []
    # corrcount = 0
    for i, frame in enumerate(frames):
        lips = extractLipsFromFrame(frame)
        if len(lips) == 0:
            print("failed to get face")
            detected.append(frame)
        else:
            detected.append(lips)
            # corrcount+=1
        cv2.imshow(str(i), detected[-1])
        # cv2.imshow(str(i), inputframe)
    # accuracy = (corrcount/len(frames))*100
    # with open('../Image-Processing-Test/ModelsTiming.csv', 'a', newline='') as f:
    #     writer = csv.writer(f)
    #     # writer.writerow(['Model', 'Video Name', 'Time Taken', 'Accuracy'])
    #     vidName = videoPath.split('/')
    #     writer.writerow(['Dlib', vidName[len(vidName)-1], time.time() - startTime, accuracy])

    print("Run Time: {} Seconds".format(time.time() - startTime))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
