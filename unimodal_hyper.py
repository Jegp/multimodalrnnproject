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

    model.add(LSTM({{choice([200, 400, 800, 1000])}}, 
              input_shape = (1, max_features), 
              return_sequences = True))

    if conditional({{choice(['two', 'three'])}}) == 'three':
        model.add(LSTM({{choice([200, 400, 800, 1000])}},
                  return_sequences = True))
        model.add(LSTM({{choice([200, 400, 800, 1000])}},
                  return_sequences = False))
    else:
        model.add(LSTM({{choice([200, 400, 800, 1000])}},
                  return_sequences = False))
    
    # Avoid overfitting by dropping data
    model.add(Dropout({{uniform(0, 1)}}))

    # Regular dense nn with sigmoid activation function
    model.add(Dense(y_train.shape[1], activation = 'softmax'))

    ## Compile model
    model.compile(
        loss='categorical_crossentropy'
      , optimizer='rmsprop'
      , metrics = ['accuracy']    # Collect accuracy metric 
    )

    ## Early stop
    early_stopping = EarlyStopping(monitor='val_loss', patience=8)

    ## Fit model
    model.fit(X_train, y_train, 
              batch_size={{choice([64, 128, 256])}}, 
              nb_epoch=500,
              validation_data=(X_test, y_test),
              callbacks=[early_stopping])

    ## Extract score)
    score, acc = model.evaluate(X_test, y_test, verbose=0)

    print("Accuracy: ", acc)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model, data=data, algo=tpe.suggest, max_evals=10, trials=Trials())
    X_train, y_train, X_test, y_test, m, n = data()
    print(best_model.evaluate(X_test, y_test))
    best_model.evaluate(X_test, y_test)
    print(best_run)

