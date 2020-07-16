""" This file contains a small demo for our project using
  only the CNN architectures. """
import glob
import time
import logging

import cv2
import numpy as np
from silence_tensorflow import silence_tensorflow

silence_tensorflow()
import tensorflow as tf
from NN_Models import AdverbModel, CharacterCNN, ColorModel, CommandModel, NumberModel, PrepositionModel
from CNN_DataPreparation import ConcatenateDataSet


def getTrainedModel(modelPath, modelCategory):
    print("Fetching model weights...")
    modelDict = {'Adverb': AdverbModel.AdverbNet(),
                 'Alphabet': CharacterCNN.CharCNN(),
                 'Colors': ColorModel.ColorsNet(),
                 'Commands': CommandModel.CommandsNet(),
                 'Numbers': NumberModel.NumbersNet(),
                 'Prepositions': PrepositionModel.PrepositionsNet()}

    model = modelDict[modelCategory]
    model.Model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    model.Model = tf.keras.models.load_model(modelPath + '/')
    print("{} Model is loaded...".format(modelCategory))
    return model.Model


def predictOneVideo(classDict, videoPath, operationMode):
    # Model Loading:
    savedModelPath = './SavedModels/'
    videoPath = videoPath.replace("\\", "/")
    path = videoPath.split('/')
    categoryCNN = path[-1].split('_')[0]
    if categoryCNN == '':
        return "Error! No videos were passed"

    savedModelPath += categoryCNN
    dictForClass = classDict[categoryCNN]
    cnnModel = getTrainedModel(savedModelPath, categoryCNN)

    # Video Concatenation Operations:
    print("Starting video preparation operations...")
    haarDetector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    videoFrames = ConcatenateDataSet.getVideoFrames(videoPath)[:30]

    if operationMode != 'SERVER':
        # Rotate Frames:
        videoFrames = [cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE) for x in videoFrames]

    videoFrames = [ConcatenateDataSet.extractLipsHaarCascade(haarDetector, x) for x in videoFrames]
    concatenatedImage = ConcatenateDataSet.stackFramesToImage(videoFrames)

    if concatenatedImage is not None:
        # Image Preparation:
        if len(concatenatedImage.shape) == 3 and concatenatedImage.shape[2] != 1:
            concatenatedImage = cv2.cvtColor(concatenatedImage, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(concatenatedImage, (224, 224))
        img = np.array(img, dtype=np.float32)
        img = np.reshape(img, (-1, 224, 224, 1))
        img = img / 255

        # Model Prediction:
        print("Starting Model Prediction...")
        modelPrediction = cnnModel.predict_classes(img)
        log.info(modelPrediction)
        log.info(cnnModel.predict(img))
        dictForClass = list(dictForClass.items())
        prediction = [item for item in dictForClass if modelPrediction[0] in item][0][0]
    else:
        prediction = "Error! Video {} has less than 30 frames".format(path[-1])
    return prediction


def createClassLabelsDict():
    """Function to create the labels class dictionary
     based on data generated from previous runs which will remain static"""
    D = {'Prepositions': {'at': 0, 'by': 1, 'in': 2, 'with': 3},
         'Numbers': {'eight': 0, 'five': 1, 'four': 2, 'nine': 3, 'one': 4,
                     'seven': 5, 'six': 6, 'three': 7, 'two': 8},
         'Commands': {'bin': 0, 'lay': 1, 'place': 2, 'set': 3},
         'Colors': {'blue': 0, 'green': 1, 'red': 2, 'white': 3},
         'Alphabet': {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7,
                      'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14,
                      'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21,
                      'w': 22, 'x': 23, 'y': 24, 'z': 25},
         'Adverb': {'again': 0, 'now': 1, 'please': 2, 'soon': 3}}

    return D


def prototypeProject(receivedFiles, operationMode):
    AllClassLabels = createClassLabelsDict()
    mylist = glob.glob(receivedFiles)
    mylist.sort(key=lambda x: x.split('_')[-1])
    resultString = []
    for video in mylist:
        resultString.append(predictOneVideo(AllClassLabels, video, operationMode))

    resultString = " ".join(resultString)
    return resultString


if __name__ == "__main__":
    start_time = time.time()

    ModeOfOperation = 'SERVER'  # Operates normally with the server.

    if ModeOfOperation == 'SERVER':
        receivedFilesFromServer = 'C:/Users/Amr Khaled/Desktop/Projects/Lipify-server/uploads/*.mp4'
        predictionFilePath = 'C:/Users/Amr Khaled/Desktop/Projects/Lipify-server/prediction.txt'
    else:
        receivedFilesFromServer = "Prototype-Test-Videos/*.mp4"
        predictionFilePath = 'Prototype-Test-Videos/Predictions.txt'

    logFilePath = predictionFilePath.split('/')[:-1]
    logFilePath = "/".join(logFilePath)
    logging.basicConfig(filename=logFilePath + '/current run.log', filemode='w', level=logging.INFO)
    log = logging.getLogger("my-logger")
    log.info("App Start")
    log.info("Mode Of Operation: {}".format(ModeOfOperation))

    # AllClassLabels = getAllClassLabels() # from classLabels import getAllClassLabels()

    result = prototypeProject(receivedFilesFromServer, ModeOfOperation)
    predictionFile = open(predictionFilePath, "w")  # write mode
    predictionFile.write(result)
    predictionFile.close()

    print("Run Time: {} Seconds".format(time.time() - start_time))
