import multiprocessing
from collections import ChainMap

from silence_tensorflow import silence_tensorflow

silence_tensorflow()
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def getClassIndex(className):
    """
    Function that returns dictionary of class indices for a certain category
    :param className: String, Name of
    :return:
    """
    train_dir = 'C:/Users/amrkh/Desktop/CNN-Training-Images/'
    trainDirectory = train_dir + className + '/'
    train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our training data
    train_data_gen = train_image_generator.flow_from_directory(batch_size=32,
                                                               directory=trainDirectory,
                                                               shuffle=True,
                                                               target_size=(224, 224),
                                                               class_mode='categorical',
                                                               color_mode='grayscale')
    classLabelsDict = {className: train_data_gen.class_indices}
    return classLabelsDict


def getAllClassLabels():
    """
    Function that returns all class indices of our dataset categories.

    :return: Dictionary of dictionaries
    """
    mylist = ['Adverb', 'Alphabet', 'Colors', 'Commands', 'Numbers', 'Prepositions']
    p = multiprocessing.Pool()
    res = p.map(getClassIndex, mylist)
    res = dict(ChainMap(*res))
    return res


if __name__ == "__main__":
    result = getAllClassLabels()
    print(result)
