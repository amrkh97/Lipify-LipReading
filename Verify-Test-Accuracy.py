from silence_tensorflow import silence_tensorflow

silence_tensorflow()
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import csv
import subprocess
import AdverbModel
import CharacterCNN
import ColorModel
import CommandModel
import NumberModel
import PrepositionModel

modelDict = {'Adverb': AdverbModel.AdverbNet(),
             'Alphabet': CharacterCNN.CharCNN(),
             'Colors': ColorModel.ColorsNet(),
             'Commands': CommandModel.CommandsNet(),
             'Numbers': NumberModel.NumbersNet(),
             'Prepositions': PrepositionModel.PrepositionsNet()}


def verifyTestCategory(categoryName='Adverb', TrainedModel=None, batch_size=16):
    test_dir = '../CNN-Test-Images/{}/'.format(categoryName)
    test_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our test data
    test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                             directory=test_dir,
                                                             shuffle=False,
                                                             target_size=(224, 224),
                                                             class_mode='categorical',
                                                             color_mode='grayscale')
    model = modelDict[categoryName]
    model.Model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    model.Model = TrainedModel
    temp = model.Model.evaluate(test_data_gen)
    return temp[1]


def verifyTrainCategory(categoryName='Adverb', batch_size=16):
    train_dir = '../CNN-Training-Images/{}/'.format(categoryName)
    train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our test data
    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                               directory=train_dir,
                                                               shuffle=False,
                                                               target_size=(224, 224),
                                                               class_mode='categorical',
                                                               color_mode='grayscale')
    model = modelDict[categoryName]
    model.Model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    modelPath = 'C:/Users/amrkh/Desktop/SavedModels/'
    modelPath += categoryName
    model.Model = tf.keras.models.load_model(modelPath + '/')
    temp = model.Model.evaluate(train_data_gen)
    return model.Model, temp[1]


if __name__ == "__main__":
    categoriesList = ['Adverb', 'Alphabet', 'Commands', 'Colors', 'Numbers', 'Prepositions']

    with open('Project_Accuracy.csv', 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile)
        filewriter.writerow(['Category', 'Train Accuracy', 'Test Accuracy'])

    for category in categoriesList:
        categoryModel, categoryTrainAccuracy = verifyTrainCategory(categoryName=category)

        categoryTrainAccuracy = str(round(categoryTrainAccuracy * 100, 2))
        categoryTestAccuracy = str(round(verifyTestCategory(categoryName=category,
                                                            TrainedModel=categoryModel) * 100, 2))
        print("Category {}: Train --> {}%, Test --> {}%".format(category, categoryTrainAccuracy,
                                                                categoryTestAccuracy))

        with open('Project_Accuracy.csv', 'a', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow([category, categoryTrainAccuracy, categoryTestAccuracy])

    subprocess.check_call(['Rscript', 'Visualize_Model_Accuracy.R'], shell=False)
