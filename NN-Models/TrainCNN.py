import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from CharacterCNN import CharCNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 32
epochs = 500
train_dir = 'D:/CNN-Training-Images/Alphabet/'
train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our training data
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=False,
                                                           target_size=(400, 400),
                                                           class_mode='categorical')

C = CharCNN((400, 400, 3), 26)
C.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="../charCNNweights.ckpt",
                                                 save_weights_only=True,
                                                 verbose=1)

history = C.model.fit(
    train_data_gen,
    steps_per_epoch=100,  # All Images present in characters dataset
    epochs=epochs,
    callbacks=[cp_callback]
)

acc = history.history['accuracy']
loss = history.history['loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.legend(loc='lower right')
plt.title('Training Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.legend(loc='upper right')
plt.title('Training Loss')
plt.show()
