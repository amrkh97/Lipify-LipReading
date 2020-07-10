""" This file contains a small demo for our project using
  only the CNN architectures. """
import glob
import time
import cv2
import numpy as np
from RGBFilterLipsDetection import GaborLauncher
from CNN_Implementation import AdverbCNN_ByHand
from CNN_DataPreparation import ConcatenateDataSet as CDS


def predictAdverbVGGVideo(videoPath):
    videoFrames = CDS.getVideoFrames(videoPath)
    print("Starting Lips Extraction ...")

    videoFrames = [GaborLauncher.startProcess(x) for x in videoFrames]
    videoFrames, _ = zip(*videoFrames)
    concatenatedImage = CDS.stackFramesToImage(videoFrames[:30])
    print("Stacked Video into Image for prediction ...")

    result = ""
    if concatenatedImage is not None:
        # Image Preparation:
        if len(concatenatedImage.shape) == 3 and concatenatedImage.shape[2] != 1:
            concatenatedImage = cv2.cvtColor(concatenatedImage, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(concatenatedImage, (224, 224))
        img = np.array(img, dtype=np.int8)

        print("Starting Adverb Model & Prediction Process ...")
        result, _ = AdverbCNN_ByHand.predictAdverb(img, "CNN-Implementation/params.pkl")

    else:
        return "Error! Video has less than 30 frames."

    return result


if __name__ == "__main__":
    start_time = time.time()

    receivedFilesFromServer = 'C:/Users/Amr Khaled/Desktop/Projects/Lipify-server/uploads/*.mp4'
    mylist = glob.glob(receivedFilesFromServer)
    mylist.sort(key=lambda x: x.split('_')[-1])
    resultString = []
    for video in mylist:
        resultString.append(predictAdverbVGGVideo(video))

    resultString = " ".join(resultString)
    print("Prediction Result: {}".format(resultString))
    predictionFilePath = 'C:/Users/Amr Khaled/Desktop/Projects/Lipify-server/prediction.txt'

    predictionFile = open(predictionFilePath, "w")  # write mode
    predictionFile.write(resultString)
    predictionFile.close()

    print("Run Time: {} Seconds".format(time.time() - start_time))
