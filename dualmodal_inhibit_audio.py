from __future__ import print_function

import numpy
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential, Model
from keras.layers import Concatenate
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.externals import joblib
from keras.utils.np_utils import to_categorical                                                                        
from keras.utils.vis_utils import model_to_dot

from precision import precision, recall, fmeasure

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

def data():
    X_train_v, X_test_v, y_train_v, y_test_v = joblib.load('305010.pkl')[0]
    X_train_a, X_test_a, y_train_a, y_test_a = joblib.load('305010.pkl')[1]

    X_train_a = numpy.array(X_train_a)
    X_train_a = X_train_a.reshape((X_train_a.shape[0], 1, X_train_a.shape[1]))
    y_train_a = to_categorical(y_train_a)

    X_test_a = numpy.array(X_test_a)
    X_test_a = X_test_a.reshape((X_test_a.shape[0], 1, X_test_a.shape[1]))
    y_test_a = to_categorical(y_test_a)

    max_features_a = X_train_a.shape[2]
    maxlen_a = X_train_a.shape[0]
    
    X_train_v = numpy.array(X_train_v)
    X_train_v = X_train_v.reshape((X_train_v.shape[0], 1, X_train_v.shape[1]))
    y_train_v = to_categorical(y_train_v)

    X_test_v = numpy.array(X_test_v)
    X_test_v = X_test_v.reshape((X_test_v.shape[0], 1, X_test_v.shape[1]))
    y_test_v = to_categorical(y_test_v)

    max_features_v = X_train_v.shape[2]
    maxlen_v = X_train_v.shape[0]

    return X_train_a, y_train_a, X_test_a, y_test_a, max_features_a, maxlen_a, X_train_v, y_train_v, X_test_v, y_test_v, max_features_v, maxlen_v

def model(X_train_a, y_train_a, X_test_a, y_test_a, max_features_a, maxlen_a, X_train_v, y_train_v, X_test_v, y_test_v, max_features_v, maxlen_v):

    model_auditory = Sequential()
    model_auditory.add(LSTM(1000, input_shape=(1, max_features_a), return_sequences=True))
    model_auditory.add(LSTM(800, return_sequences=False))
    model_auditory.add(Dropout(0.8)) # Remove part of the auditory data

    model_visual = Sequential()
    model_visual.add(LSTM(210, input_shape=(1, max_features_v), return_sequences=True))
    model_visual.add(LSTM(120, return_sequences=False))

    ## Merge models
    ## - Sequential cannot be used to concatenate, so we have to use the functional API
    out = Concatenate()([model_auditory.output, model_visual.output])

    # Avoid overfitting
    out = Dropout(0.5)(out)
    # Regular dense nn with sigmoid activation function
    out = Dense(y_train_a.shape[1], activation='softmax')(out)

    model = Model(inputs = [model_auditory.input, model_visual.input], outputs = out)
 
    ## Compile model
    model.compile(
        loss='categorical_crossentropy'
      , optimizer='rmsprop'
      , metrics = ['accuracy', precision, recall, fmeasure]    # Collect accuracy metric 
    )

    ## Early stop
    early_stopping = EarlyStopping(monitor='val_loss', patience=16)

    ## Fit model
    history = model.fit([X_train_a, X_train_v], y_train_a, 
              batch_size=512,
              epochs=50,
              validation_data=([X_test_a, X_test_v], y_test_a),
              callbacks=[early_stopping])

    # summarize history for accuracy
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['val_loss'])
    plt.title('Audio and video model, inhibited audio')
    plt.xlabel('Epoch')
    plt.legend(['Accuracy', 'Loss'], loc='upper right')
    plt.savefig('model_dualmodal_inhibit_audio.png')

if __name__ == '__main__':
    X_train_a, y_train_a, X_test_a, y_test_a, max_features_a, maxlen_a, X_train_v, y_train_v, X_test_v, y_test_v, max_features_v, maxlen_v = data()
    model(X_train_a, y_train_a, X_test_a, y_test_a, max_features_a, maxlen_a, X_train_v, y_train_v, X_test_v, y_test_v, max_features_v, maxlen_v)
