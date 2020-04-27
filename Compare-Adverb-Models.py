"""The file serves as a comparator for various implementations of the adverb model"""
import pickle
import time

import numpy as np
from silence_tensorflow import silence_tensorflow
from tqdm import tqdm

silence_tensorflow()
import tensorflow as tf
import AdverbModel
import Adverb_VGG
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import extract_labels, extract_data, predict


def evaluateAdverbCNN_TF(test_set_path, model_path):
    adverbModel = AdverbModel.AdverbNet()
    adverbModel.Model = tf.keras.models.load_model(model_path)
    test_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our test data
    test_data_gen = test_image_generator.flow_from_directory(batch_size=16,
                                                             directory=test_set_path,
                                                             shuffle=False,
                                                             target_size=(224, 224),
                                                             class_mode='categorical',
                                                             color_mode='grayscale')
    val_acc = adverbModel.Model.evaluate(test_data_gen)
    print("Adverb CNN -using TF- Accuracy: {}%".format(val_acc[1] * 100))
    return round(val_acc[1] * 100, 2)


def evaluateAdverbCNN_NP(test_set_path, model_path, number_of_test_images=50):
    model_path += 'params.pkl'
    params, cost = pickle.load(open(model_path, 'rb'))
    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    if number_of_test_images > 3000:
        return -1

    testImages_path = test_set_path + 'Adverb-test-images-idx3-ubyte.gz'
    testLabels_path = test_set_path + 'Adverb-test-labels-idx1-ubyte.gz'

    X = extract_data(testImages_path, number_of_test_images, 224)
    y_dash = extract_labels(testLabels_path, number_of_test_images).reshape(number_of_test_images, 1)
    # Normalize the data
    X -= int(np.mean(X))  # subtract mean
    X /= int(np.std(X))  # divide by standard deviation
    test_data = np.hstack((X, y_dash))

    X = test_data[:, 0:-1]
    X = X.reshape(len(test_data), 1, 224, 224)
    y = test_data[:, -1]
    corr = 0

    t = tqdm(range(len(X)), leave=True)
    for i in t:
        x = X[i]
        pred, prob = predict(x, f1, f2, w3, w4, b1, b2, b3, b4)
        if pred == int(y[i]):
            corr += 1
        t.set_description("Acc:%0.2f%%" % (float(corr / (i + 1)) * 100))

    modelAccuracy = float(corr / len(test_data) * 100)
    print("Adverb CNN -using VGG- Accuracy: {}%".format(modelAccuracy))
    return round(modelAccuracy, 2)


def evaluateAdverbCNN_VGG(test_set_path, model_path):
    adverbModel = Adverb_VGG.AdverbVGG()
    adverbModel.Model = tf.keras.models.load_model(model_path)
    test_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our test data
    test_data_gen = test_image_generator.flow_from_directory(batch_size=16,
                                                             directory=test_set_path,
                                                             shuffle=False,
                                                             target_size=(224, 224),
                                                             class_mode='categorical')
    val_acc = adverbModel.Model.evaluate(test_data_gen)
    print("Adverb CNN -using VGG- Accuracy: {}%".format(val_acc[1] * 100))
    return round(val_acc[1] * 100, 2)


if __name__ == "__main__":
    start_time = time.time()
    commonPath = 'C:/Users/amrkh/Desktop/'
    accuracyDict = {}

    # Adverb CNN - Custom Architecture -:
    CNN_Custom_testSet = commonPath + 'CNN-Test-Images/Adverb/'
    CNN_Custom_savedModel = commonPath + 'SavedModels/Adverb/'

    accuracyDict['Custom_Arch'] = evaluateAdverbCNN_TF(CNN_Custom_testSet, CNN_Custom_savedModel)

    # Adverb CNN - VGG PreTrained -:
    CNN_VGG_testSet = commonPath + 'CNN-Test-Images/Adverb/'
    CNN_VGG_savedModel = commonPath + 'SavedModels/VGG/Adverb/'

    accuracyDict['VGG'] = evaluateAdverbCNN_VGG(CNN_VGG_testSet, CNN_VGG_savedModel)

    # Adverb CNN - Numpy Implementation -:
    CNN_NP_testSet = commonPath + 'Lipify-LipReading/Compressed-Dataset/'
    CNN_NP_savedModel = commonPath + 'Lipify-LipReading/CNN-Implementation/'
    numberOfTestExample = 10
    accuracyDict['Numpy'] = evaluateAdverbCNN_NP(CNN_NP_testSet, CNN_NP_savedModel, numberOfTestExample)

    print("Different Models Accuracy:")
    print(accuracyDict)
    runTime = round(time.time() - start_time, 2)
    print("Run Time: {} Seconds".format(runTime))
