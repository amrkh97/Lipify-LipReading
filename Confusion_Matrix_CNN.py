import os

from silence_tensorflow import silence_tensorflow

from NN_Models import AdverbModel, PrepositionModel, ColorModel, CommandModel, NumberModel, CharacterCNN

silence_tensorflow()
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":

    dirName = 'Project_Insights/Model_Confusion_Matrix'
    if not os.path.exists(dirName):
        os.makedirs(dirName)

    if not os.path.isfile('file_path'):
        df = pd.DataFrame(list())
        df.to_csv('Project_Insights/Model_Classification_Report.csv')

    adverb_names = ['again', 'now', 'please', 'soon']
    colors_names = ['blue', 'green', 'red', 'white']
    commands_names = ['bin', 'lay', 'place', 'set']
    prepositions_names = ['at', 'by', 'in', 'with']
    numbers_names = ['eight', 'five', 'four', 'nine', 'one',
                     'seven', 'six', 'three', 'two']
    alphabet_names = [chr(x) for x in range(ord('a'), ord('z') + 1)]
    alphabet_names.remove('w')

    listDict = {'Adverb': adverb_names,
                'Alphabet': alphabet_names,
                'Colors': colors_names,
                'Commands': commands_names,
                'Numbers': numbers_names,
                'Prepositions': prepositions_names}

    common_path = 'C:/Users/amrkh/Desktop/'

    modelDict = {'Adverb': AdverbModel.AdverbNet(),
                 'Alphabet': CharacterCNN.CharCNN(),
                 'Colors': ColorModel.ColorsNet(),
                 'Commands': CommandModel.CommandsNet(),
                 'Numbers': NumberModel.NumbersNet(),
                 'Prepositions': PrepositionModel.PrepositionsNet()}

    categoriesList = ['Adverb', 'Alphabet', 'Commands', 'Colors', 'Numbers', 'Prepositions']
    # categoriesList = ['Adverb', ]

    for category in categoriesList:
        test_dir = common_path + 'CNN-Test-Images/{}/'.format(category)
        checkpoint_path = common_path + 'SavedModels/{}/'.format(category)

        test_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our test data
        test_data_gen = test_image_generator.flow_from_directory(batch_size=16,
                                                                 directory=test_dir,
                                                                 shuffle=False,
                                                                 target_size=(224, 224),
                                                                 class_mode='categorical',
                                                                 color_mode='grayscale')

        model = modelDict[category].Model
        model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
        model = tf.keras.models.load_model(checkpoint_path)
        Y_pred = model.predict(test_data_gen)
        y_pred = np.argmax(Y_pred, axis=1)
        conf_mat = confusion_matrix(test_data_gen.classes, y_pred)
        target_names = listDict[category]

        report = classification_report(test_data_gen.classes, y_pred,
                                       output_dict=True, target_names=target_names)
        report = pd.DataFrame(report).transpose()
        report.to_csv('Project_Insights/Model_Classification_Report.csv', mode='a', header=True)

        # Save Figure to png:
        plt.figure()
        sns.heatmap(conf_mat, annot=True, xticklabels=target_names, cbar=False,
                    yticklabels=target_names, fmt='d', cmap="Blues")

        plt.xlabel('Predicted Labels')
        plt.ylabel('Actual Labels')
        plt.savefig('Project_Insights/Model_Confusion_Matrix/{}.png'.format(category))
