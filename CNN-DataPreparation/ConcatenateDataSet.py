import glob
import os
import time

import cv2
import numpy as np

from extract_lips import extractLipsFromFrame
from frameManipulator import FPS, getVideoDataFromPath

commands = ['bin', 'lay', 'place', 'set']
prepositions = ['at', 'by', 'in', 'with']
colors = ['blue', 'green', 'red', 'white']
adverbs = ['again', 'now', 'please', 'soon']
alphabet = [chr(x) for x in range(ord('a'), ord('z') + 1)]
numbers = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
categories = ['Adverb', 'Alphabet', 'Commands', 'Colors', 'Numbers', 'Prepositions']
commonCNNDataPath = 'D:/CNN-Training-Images/'


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
    if len(listOfFrames) < FPS:
        return None
    newList = [np.hstack(listOfFrames[:5]), np.hstack(listOfFrames[5:10]), np.hstack(listOfFrames[10:15]),
               np.hstack(listOfFrames[15:20]), np.hstack(listOfFrames[20:25]), np.hstack(listOfFrames[25:30])]
    return np.vstack(newList)


def saveImage(image, imagePath):
    """Function to save an image in grayscale to a specific path"""
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


def prepareSingleVideoForCNN(path):
    """Function to prepare a single video to be redy for CNN training"""
    vidData = getVideoDataFromPath(path)
    videoFrames = getVideoFrames(path)
    videoFrames = [extractLipsFromFrame(x) for x in videoFrames]

    if len(videoFrames) != 0:
        stackedImage = stackFramesToImage(videoFrames)
        videoLabel = vidData.identifier.split('_')[0]
        imageSavePath = commonCNNDataPath + vidData.category + '/{}'.format(videoLabel)
        saveImage(stackedImage, imageSavePath)
    else:
        print("Error in finding video with path: {}".format(path))


def prepareDataSetForCNN(Number_Of_Speakers):
    """Function that traverses the whole dataset and creates new directory for the CNN"""
    for i in range(Number_Of_Speakers):
        for category in categories:
            videoPath = "../New-DataSet-Videos/S{}/{}/".format(i + 1, category) + "*.mp4"
            vidList = glob.glob(videoPath)

            def f(x):
                return x.replace("\\", '/')

            vidList = [f(x) for x in vidList]
            for j in vidList:
                prepareSingleVideoForCNN(j)
        print("Finished Speaker {}".format(i + 1))


if "__main__" == __name__:
    startTime = time.time()
    numberSpeakers = 20

    createCNNDataDirectories()
    prepareDataSetForCNN(numberSpeakers)

    print("Finished preparing the videos in {} seconds".format(time.time() - startTime))
