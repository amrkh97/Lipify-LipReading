import os
from silence_tensorflow import silence_tensorflow

silence_tensorflow()
from tensorflow.keras.utils import plot_model
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


def drawModel(model, fileName):
    plot_model(model, show_shapes=True, show_layer_names=False,
               to_file='Model_Graphs/{}_CNN.png'.format(fileName))


if __name__ == "__main__":

    dirName = 'Model_Graphs'
    if not os.path.exists(dirName):
        os.makedirs(dirName)

    categoriesList = ['Adverb', 'Alphabet', 'Commands', 'Colors', 'Numbers', 'Prepositions']
    for category in categoriesList:
        CNN_Model = modelDict[category].Model
        drawModel(CNN_Model, category)
