from silence_tensorflow import silence_tensorflow

silence_tensorflow()
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Input, GlobalAveragePooling2D
from tensorflow_core.python.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

VGG19_MODEL = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')
VGG19_MODEL.trainable = False


class AdverbVGG(object):
    def __init__(self):
        self.Model = Sequential()
        self.build()

    def build(self):
        self.Model.add(Input(name='the_input', shape=(224, 224, 3), batch_size=16, dtype='float32'))

        self.Model.add(VGG19_MODEL)
        self.Model.add(GlobalAveragePooling2D())
        self.Model.add(Dense(4, kernel_initializer='he_normal', name='dense-2'))
        self.Model.add(Activation('softmax', name='softmax'))

    def summary(self):
        self.Model.summary()


if __name__ == "__main__":
    common_path = 'C:/Users/amrkh/Desktop/'
    C = AdverbVGG()
    C.Model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])
    C.Model.summary()

    with tf.device('/device:GPU:0'):
        batch_size = 16
        epochs = 3
        train_dir = common_path + 'CNN-Training-Images/Adverb/'
        test_dir = common_path + 'CNN-Test-Images/Adverb/'
        checkpoint_path = common_path + 'SavedModels/VGG/Adverb/'

        train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our training data
        train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                                   directory=train_dir,
                                                                   shuffle=True,
                                                                   target_size=(224, 224),
                                                                   class_mode='categorical')

        test_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our test data
        test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                                 directory=test_dir,
                                                                 shuffle=False,
                                                                 target_size=(224, 224),
                                                                 class_mode='categorical')

        # C.Model = tf.keras.models.load_model(checkpoint_path)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                    patience=3,
                                                    restore_best_weights=True,
                                                    baseline=0.25)

        history = C.Model.fit(train_data_gen,
                              steps_per_epoch=3564,  # Number of images // Batch size
                              epochs=epochs,
                              verbose=1,
                              validation_data=test_data_gen,
                              validation_steps=187,
                              callbacks=[callback])

        C.Model.save(checkpoint_path, save_format='tf')
        # Evaluate Model:
        # Accuracy: 28.94%
        C.Model.evaluate(test_data_gen)
