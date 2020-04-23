from silence_tensorflow import silence_tensorflow

silence_tensorflow()
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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


def verifyCategory(categoryName='Adverb', batch_size=16):
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
    modelPath = 'C:/Users/amrkh/Desktop/SavedModels/'
    modelPath += categoryName
    model.Model = tf.keras.models.load_model(modelPath + '/')
    temp = model.Model.evaluate(test_data_gen)
    return temp[1]


if __name__ == "__main__":
    categoriesList = ['Adverb', 'Alphabet', 'Commands', 'Colors', 'Numbers', 'Prepositions']
    lossPerCategory = []
    for category in categoriesList:
        categoryAccuracy = str(round(verifyCategory(categoryName=category) * 100, 2))
        lossPerCategory.append({category: categoryAccuracy + '%'})
    print(lossPerCategory)
