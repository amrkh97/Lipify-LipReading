import glob
import os
import time

import cv2
import numpy as np

from Pre_Processing import frameManipulator

commands = ['bin', 'lay', 'place', 'set']
prepositions = ['at', 'by', 'in', 'with']
colors = ['blue', 'green', 'red', 'white']
adverbs = ['again', 'now', 'please', 'soon']
alphabet = [chr(x) for x in range(ord('a'), ord('z') + 1)]
numbers = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
categories = ['Adverb', 'Alphabet', 'Commands', 'Colors', 'Numbers', 'Prepositions']
commonCNNDataPath = 'D:/CNN-Test-Images/'


def getVideoFrames(videoPath):
    """Function to return a video's frames in a list"""
    vidcap = cv2.VideoCapture(videoPath)
    success, image = vidcap.read()
    allFrames = []
    while success:
        allFrames.append(image)
        success, image = vidcap.read()
    return allFrames


def stackFramesToImage(listOfFrames):
    """Function to concat frames into a single picture"""
    if len(listOfFrames) < frameManipulator.FPS:
        return None
    newList = [np.hstack(listOfFrames[:5]), np.hstack(listOfFrames[5:10]), np.hstack(listOfFrames[10:15]),
               np.hstack(listOfFrames[15:20]), np.hstack(listOfFrames[20:25]), np.hstack(listOfFrames[25:30])]
    return np.vstack(newList)


def saveImage(image, imagePath):
    """Function to save an image in grayscale to a specific path"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    index = len(os.listdir(imagePath))
    imagePath = imagePath + '/{}.jpg'.format(index)
    cv2.imwrite(imagePath, image)


def createCNNDataDirectories():
    """Function to create label directories for each category for training the CNN"""
    for command in commands:
        dirName = commonCNNDataPath + '/Commands/{}/'.format(command)
        if not os.path.exists(dirName):
            os.makedirs(dirName)

    for preposition in prepositions:
        dirName = commonCNNDataPath + '/Prepositions/{}/'.format(preposition)
        if not os.path.exists(dirName):
            os.makedirs(dirName)

    for color in colors:
        dirName = commonCNNDataPath + '/Colors/{}/'.format(color)
        if not os.path.exists(dirName):
            os.makedirs(dirName)

    for adverb in adverbs:
        dirName = commonCNNDataPath + '/Adverb/{}/'.format(adverb)
        if not os.path.exists(dirName):
            os.makedirs(dirName)

    for letter in alphabet:
        dirName = commonCNNDataPath + '/Alphabet/{}/'.format(letter)
        if not os.path.exists(dirName):
            os.makedirs(dirName)

    for number in numbers:
        dirName = commonCNNDataPath + '/Numbers/{}/'.format(number)
        if not os.path.exists(dirName):
            os.makedirs(dirName)


def extractLipsHaarCascade(haarDetector, frame):
    """Function to extract lips from a frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray = 0
    faces = haarDetector.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        roi_gray = cv2.resize(gray, (150, 100))
        return roi_gray

    for (x, y, w, h) in faces:
        roi_gray = gray[y + (2 * h // 3):y + h, x:x + w]
    roi_gray = cv2.resize(roi_gray, (150, 100))
    return roi_gray


def prepareSingleVideoForCNN(path, haarDetector):
    """Function to prepare a single video to be redy for CNN training"""
    vidData = frameManipulator.getVideoDataFromPath(path)
    videoFrames = getVideoFrames(path)
    videoFrames = [extractLipsHaarCascade(haarDetector, x) for x in videoFrames]

    if len(videoFrames) != 0:
        stackedImage = stackFramesToImage(videoFrames)
        videoLabel = vidData.identifier.split('_')[0]
        imageSavePath = commonCNNDataPath + vidData.category + '/{}'.format(videoLabel)
        saveImage(stackedImage, imageSavePath)
    else:
        print("Error in finding video with path: {}".format(path))


def prepareDataSetForCNN(firstSpeaker, secondSpeaker):
    """Function that traverses the whole dataset and creates new directory for the CNN"""
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    for i in range(firstSpeaker, secondSpeaker):
        for category in categories:
            sTime = time.time()
            videoPath = "../New-DataSet-Videos/S{}/{}/".format(i, category) + "*.mp4"
            vidList = glob.glob(videoPath)

            def f(x):
                return x.replace("\\", '/')

            vidList = [f(x) for x in vidList]
            for j in vidList:
                prepareSingleVideoForCNN(j, detector)

            print("Finished category : {}, for speaker: {}".format(category, i))
            print("In:{} Seconds".format(time.time() - sTime))

        print("Finished Speaker {}".format(i))


def main():
    startTime = time.time()
    firstSpeaker = 23
    secondSpeaker = 24
    createCNNDataDirectories()
    prepareDataSetForCNN(firstSpeaker, secondSpeaker)

    print("Finished preparing the videos in {} seconds".format(time.time() - startTime))


if __name__ == "__main__":
    main()
