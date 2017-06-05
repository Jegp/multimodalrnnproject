from __future__ import print_function

import numpy
import numpy as np
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.externals import joblib
from keras.utils.np_utils import to_categorical                                                                        

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

import matplotlib.pyplot as plt

def data():
    X_train, X_test, y_train, y_test = joblib.load('305010.pkl')[0]

    X_train = numpy.array(X_train)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    y_train = to_categorical(y_train)

    X_test = numpy.array(X_test)
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    y_test = to_categorical(y_test)

    max_features = X_train.shape[2]
    maxlen = X_train.shape[0]
    
    return X_train, y_train, X_test, y_test, max_features, maxlen

def model(X_train, y_train, X_test, y_test, max_features, maxlen):
    model = Sequential()

    model.add(LSTM(210, 
              input_shape = (1, max_features), 
              return_sequences = True))

    model.add(LSTM(120,
          return_sequences = False))
    
    # Avoid overfitting by dropping data
    model.add(Dropout(0.651797))

    # Regular dense nn with sigmoid activation function
    model.add(Dense(y_train.shape[1], activation = 'softmax'))
 
    ## Compile model
    model.compile(
        loss='categorical_crossentropy'
      , optimizer='rmsprop'
      , metrics = ['accuracy']    # Collect accuracy metric 
    )

    ## Early stop
    early_stopping = EarlyStopping(monitor='val_loss', patience=32)

    ## Fit model
    history = model.fit(X_train, y_train, 
              batch_size=256, 
              epochs=500,
              validation_data=(X_test, y_test),
              callbacks=[early_stopping])

    # summarize history for accuracy
    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Audio model')
    plt.xlabel('Epoch')
    plt.legend(['Accuracy, train', 'Accuracy, test', 'Loss, train', 'Loss, test'], loc='upper right')
    plt.savefig('model_video.png')

if __name__ == '__main__':
    X_train, y_train, X_test, y_test, m, n = data()
    model(X_train, y_train, X_test, y_test, m, n)

