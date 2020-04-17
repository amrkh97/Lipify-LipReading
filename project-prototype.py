""" This file contains a small demo for our project using
  only the CNN architectures. """
import multiprocessing
from itertools import product

import cv2
import numpy as np
import tensorflow as tf

import AdverbModel
import CharacterCNN
import ColorModel
import CommandModel
import NumberModel
import PrepositionModel
from ConcatenateDataSet import getVideoFrames, extractLipsHaarCascade, stackFramesToImage
from classLabels import getAllClassLabels
import time


def getTrainedModel(modelPath, modelCategory):
    modelDict = {'Adverb': AdverbModel.AdverbNet(),
                 'Alphabet': CharacterCNN.CharCNN(),
                 'Colors': ColorModel.ColorsNet(),
                 'Commands': CommandModel.CommandsNet(),
                 'Numbers': NumberModel.NumbersNet(),
                 'Prepositions': PrepositionModel.PrepositionsNet()}

    model = modelDict[modelCategory]
    model.Model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    model.Model = tf.keras.models.load_model(modelPath + '/')
    return model.Model


def predictOneVideo(classDict, videoPath):
    # Model Loading:
    savedModelPath = 'C:/Users/amrkh/Desktop/SavedModels/'
    path = videoPath.split('/')
    categoryCNN = path[-1].split('_')[0]
    savedModelPath += categoryCNN
    dictForClass = classDict[categoryCNN]
    cnnModel = getTrainedModel(savedModelPath, categoryCNN)

    # Video Concatenation Operations:
    haarDetector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    videoFrames = getVideoFrames(videoPath)
    videoFrames = [extractLipsHaarCascade(haarDetector, x) for x in videoFrames]
    concatenatedImage = stackFramesToImage(videoFrames)

    # Image Preparation:
    if len(concatenatedImage.shape) == 3 and concatenatedImage.shape[2] != 1:
        concatenatedImage = cv2.cvtColor(concatenatedImage, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(concatenatedImage, (224, 224))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, 224, 224, 1))
    img = img / 255

    # Model Prediction:
    modelPrediction = cnnModel.predict_classes(img)
    dictForClass = list(dictForClass.items())
    prediction = [item for item in dictForClass if modelPrediction[0] in item][0][0]

    return prediction


if __name__ == "__main__":
    start_time = time.time()
    AllClassLabels = getAllClassLabels()
    mylist = ['Colors_Blue.mp4', ]
    # 'Colors_Blue.mp4']  # List of video paths to predict

    with multiprocessing.Pool(processes=len(mylist)) as threadPool:
        result = threadPool.starmap(predictOneVideo, product([AllClassLabels, ], mylist))

    print(result)
    print("Run Time: {} Seconds".format(time.time() - start_time))
