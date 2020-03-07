import os

from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Conv2D, ZeroPadding2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CharCNN(object):
    def __init__(self, input_data, output_size):
        self.input_shape = input_data  # Dimension
        self.output_size = output_size  # Number of labels
        self.build()

    def build(self):
        self.input_data = Input(name='the_input', shape=self.input_shape, batch_size=32, dtype='float32')

        # self.zero_1 = ZeroPadding2D(padding=(2, 2), name='zero_1')(self.input_data)
        self.conv_1 = Conv2D(256, (3, 5), strides=(2, 2), activation='relu', kernel_initializer='he_normal',
                             name='conv_1')(self.input_data)
        self.maxp_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_1')(self.conv_1)

        self.zero_2 = ZeroPadding2D(padding=(2, 2), name='zero_2')(self.maxp_1)
        self.conv_2 = Conv2D(128, (3, 5), strides=(2, 2), activation='relu', kernel_initializer='he_normal',
                             name='conv_2')(self.zero_2)
        self.maxp_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_2')(self.conv_2)

        self.conv_3 = Conv2D(64, (3, 5), strides=(2, 2), activation='relu', kernel_initializer='he_normal',
                             name='conv_3')(self.maxp_2)
        self.maxp_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_3')(self.conv_3)

        self.resh1 = Flatten()(self.maxp_3)
        self.dense_1 = Dense(1024, kernel_initializer='he_normal', name='dense-1')(self.resh1)
        self.relu_1 = Activation('relu', name='Relu-2')(self.dense_1)
        self.drop_4 = Dropout(0.5, name='Dropout-2')(self.relu_1)

        self.dense_2 = Dense(self.output_size, kernel_initializer='he_normal', name='dense-2')(self.drop_4)
        self.labelPrediction = Activation('softmax', name='softmax')(self.dense_2)

        self.model = Model(inputs=self.input_data, outputs=self.labelPrediction)

    def summary(self):
        Model(inputs=self.input_data, outputs=self.labelPrediction).summary()


if "__main__" == __name__:
    C = CharCNN((700, 700, 3), 26)
    C.summary()



