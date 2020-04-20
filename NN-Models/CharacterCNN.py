import os

import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Conv2D, \
    MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel('ERROR')


class CharCNN(object):
    def __init__(self):
        self.Model = Sequential()
        self.build()

    def build(self):
        self.Model.add(Input(name='the_input', shape=(224, 224, 1), batch_size=16, dtype='float32'))
        self.Model.add(Conv2D(32, (3, 3), activation='sigmoid', name='conv1'))
        self.Model.add(MaxPooling2D(pool_size=(2, 2)))
        self.Model.add(Conv2D(32, (3, 3), activation='sigmoid', name='conv2'))
        self.Model.add(MaxPooling2D(pool_size=(2, 2)))
        self.Model.add(Conv2D(64, (3, 3), activation='relu', name='conv3'))
        self.Model.add(MaxPooling2D(pool_size=(2, 2)))
        self.Model.add(Flatten())

        self.Model.add(Dense(1024))
        self.Model.add(Dropout(0.5))
        self.Model.add(BatchNormalization(scale=False))
        self.Model.add(Activation('relu'))
        self.Model.add(Dropout(0.5))
        self.Model.add(Dense(26, activation='softmax'))

    def summary(self):
        self.Model.summary()


if __name__ == "__main__":
    common_path = 'C:/Users/amrkh/Desktop/'
    C = CharCNN()
    C.Model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    C.Model.summary()

    with tf.device('/device:GPU:0'):
        batch_size = 16
        epochs = 3
        train_dir = common_path + 'CNN-Training-Images/Alphabet/'
        test_dir = common_path + 'CNN-Test-Images/Alphabet/'
        checkpoint_path = common_path + 'SavedModels/Alphabet/'
        train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our training data
        train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                                   directory=train_dir,
                                                                   shuffle=True,
                                                                   target_size=(224, 224),
                                                                   class_mode='categorical',
                                                                   color_mode='grayscale')

        test_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our test data
        test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                                 directory=test_dir,
                                                                 shuffle=False,
                                                                 target_size=(224, 224),
                                                                 class_mode='categorical',
                                                                 color_mode='grayscale')

        C.Model = tf.keras.models.load_model(checkpoint_path)

        history = C.Model.fit(train_data_gen,
                              steps_per_epoch=3562,  # Number of images // Batch size
                              epochs=epochs,
                              verbose=1,
                              validation_data=test_data_gen,
                              validation_steps=187)

        # C.Model.save(checkpoint_path, save_format='tf')
