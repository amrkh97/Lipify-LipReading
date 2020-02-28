import os
import cv2
import time
import glob


def getNumberFramesPerVideo(videoPath):
    """Function to get number of frames in a specific video """
    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        print("Error in opening file at path: {}".format(videoPath))
        return -1
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


def getCategoryMaxFrames(Number_Of_Speakers, categoryName):
    """Function to get max number of frames in a video for a certain category """
    maxNumberFrames = -1
    for i in range(Number_Of_Speakers):
        videoPath = "../Videos-After-Extraction/S{}/{}/".format(i + 1, categoryName) + "*.mp4"
        videosGen = glob.iglob(videoPath)
        numberOfVideos = len(os.listdir(videoPath.split('*')[0]))
        try:
            for j in range(numberOfVideos):
                py = next(videosGen)
                maxNumberFrames = max(maxNumberFrames, getNumberFramesPerVideo(py))
        except StopIteration:
            print("Finished getting number of frames.")
    return [categoryName, maxNumberFrames]


def getDatasetMaxFrames(Number_Of_Speakers):
    """Function that returns a dictionary containing maximum number of frames for each category in whole dataset"""
    dataSetMaxFrames = {}
    adverbFrames = getCategoryMaxFrames(Number_Of_Speakers, "Adverb")
    alphabetFrames = getCategoryMaxFrames(Number_Of_Speakers, "Alphabet")
    colorsFrames = getCategoryMaxFrames(Number_Of_Speakers, "Colors")
    commandsFrames = getCategoryMaxFrames(Number_Of_Speakers, "Commands")
    numbersFrames = getCategoryMaxFrames(Number_Of_Speakers, "Numbers")
    prepositionsFrames = getCategoryMaxFrames(Number_Of_Speakers, "Prepositions")

    dataSetMaxFrames[adverbFrames[0]] = adverbFrames[1]
    dataSetMaxFrames[alphabetFrames[0]] = alphabetFrames[1]
    dataSetMaxFrames[colorsFrames[0]] = colorsFrames[1]
    dataSetMaxFrames[commandsFrames[0]] = commandsFrames[1]
    dataSetMaxFrames[numbersFrames[0]] = numbersFrames[1]
    dataSetMaxFrames[prepositionsFrames[0]] = prepositionsFrames[1]

    return dataSetMaxFrames


'''
{'Adverb': 32, 'Alphabet': 27, 'Colors': 23, 'Commands': 33, 'Numbers': 19, 'Prepositions': 23}
'''

if __name__ == "__main__":
    start_time = time.time()
    print(getDatasetMaxFrames(20))
    print("Run Time: {} Seconds.".format(time.time() - start_time))
