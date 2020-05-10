import os
from array import array
from random import shuffle

from PIL import Image


def compressOneImage(image):
    data_image = array('B')

    Im = image
    Im = Im.resize((224, 224), resample=0)
    pixel = Im.load()

    width, height = Im.size

    for x in range(0, width):
        for y in range(0, height):
            data_image.append(pixel[y, x])

    hexval = "{0:#0{1}x}".format(1, 6)  # number of files in HEX

    header = array('B')
    header.extend([0, 0, 8, 1, 0, 0])
    header.append(int('0x' + hexval[2:][:2], 16))
    header.append(int('0x' + hexval[2:][2:], 16))

    if max([width, height]) <= 256:
        header.extend([0, 0, 0, width, 0, 0, 0, height])
    else:
        raise ValueError('Image exceeds maximum size: 256x256 pixels')

    header[3] = 3  # Changing MSB for image data (0x00000803)

    data_image = header + data_image

    output_file = open('Test-image-idx3-ubyte', 'wb')
    data_image.tofile(output_file)
    output_file.close()
    os.system('gzip ' + 'Test-image-idx3-ubyte')


def compressTrainData(Names):
    for name in Names:

        data_image = array('B')
        data_label = array('B')

        FileList = []
        for dirname in os.listdir(name[0]):
            path = os.path.join(name[0], dirname)
            for filename in os.listdir(path):
                if filename.endswith(".jpg"):
                    FileList.append(os.path.join(name[0], dirname, filename))

        shuffle(FileList)

        for filename in FileList:
            filename = filename.replace("\\", "/")
            print(filename.split('/'))
            label = int(filename.split('/')[3])

            Im = Image.open(filename)
            Im = Im.resize((224, 224), resample=0)
            pixel = Im.load()

            width, height = Im.size

            for x in range(0, width):
                for y in range(0, height):
                    data_image.append(pixel[y, x])

            data_label.append(label)

        hexval = "{0:#0{1}x}".format(len(FileList), 6)  # number of files in HEX

        header = array('B')
        header.extend([0, 0, 8, 1, 0, 0])
        header.append(int('0x' + hexval[2:][:2], 16))
        header.append(int('0x' + hexval[2:][2:], 16))

        data_label = header + data_label

        # additional header for images array

        if max([width, height]) <= 256:
            header.extend([0, 0, 0, width, 0, 0, 0, height])
        else:
            raise ValueError('Image exceeds maximum size: 256x256 pixels');

        header[3] = 3  # Changing MSB for image data (0x00000803)

        data_image = header + data_image

        output_file = open(name[1] + '-images-idx3-ubyte', 'wb')
        data_image.tofile(output_file)
        output_file.close()

        output_file = open(name[1] + '-labels-idx1-ubyte', 'wb')
        data_label.tofile(output_file)
        output_file.close()

    for name in Names:
        os.system('gzip ' + name[1] + '-images-idx3-ubyte')
        os.system('gzip ' + name[1] + '-labels-idx1-ubyte')


if __name__ == "__main__":
    DirectorySaving = [['D:/CNN-Training-Images/Adverb', 'adverb'], ]
    compressTrainData(DirectorySaving)
