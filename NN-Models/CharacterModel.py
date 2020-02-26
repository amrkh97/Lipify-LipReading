import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation, Dropout, Input
from tensorflow.keras.models import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CharacterNet(object):
    def __init__(self, input_data, output_size):
        self.input_data = input_data   # Dimension
        self.output_size = output_size  # Number of labels
        self.build()

    def build(self):
        
        self.adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        
        self.inputLayer = Input(name='the_input', shape=self.input_data, dtype='float32')
        self.dense_1 = Dense(100, name='Dense-1')(self.inputLayer)
        self.relu_1 = Activation('relu', name='Relu-1')(self.dense_1)
        self.drop_1 = Dropout(0.5, name='Dropout-1')(self.relu_1)
        self.dense_2 = Dense(50, name='Dense-2')(self.drop_1)
        self.relu_2 = Activation('relu', name='Relu-2')(self.dense_2)
        self.drop_2 = Dropout(0.5, name='Dropout-2')(self.relu_2)
        self.dense_3 = Dense(self.output_size, name='Dense-3')(self.drop_2)
        self.labelPrediction = Activation('softmax', name='softmax')(self.dense_3)

        self.model = Model(inputs=self.inputLayer, outputs=self.labelPrediction)
       
    def summary(self):
        Model(inputs=self.inputLayer, outputs=self.labelPrediction).summary()


'''

################################### Encoding Trial #########################################

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

Y = ['one', 'two', 'three', 'four']
#encoder = LabelEncoder()
encoded_y = LabelEncoder().fit_transform(Y)
print(encoded_y)
dummy_y = to_categorical(encoded_y,4)
print(dummy_y)

###################################### TEST CODE ###########################################
import pandas as pd
import tensorflow as tf
 
chunk = pd.read_csv('image_label_pair.csv') #Dummy CSV for testing
chunk = chunk.values
X = chunk[:,:17].astype(float)
Y = chunk[:,17]

C = CharacterNet(17,29)
C.summary()
C.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
C.model.fit(X, Y, batch_size=10000, epochs=200,verbose=1)

print(tf.config.experimental.list_physical_devices('GPU')) # NO GPU IDENTIFIED

'''

