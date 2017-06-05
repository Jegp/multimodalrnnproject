from __future__ import print_function

import numpy
import numpy as np
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
    model_auditory.add(LSTM(800, return_sequences=True))
    #model_auditory.add(Dropout({{uniform(0, 1)}}))
    #model_auditory.add(Dense(y_test_a.shape[1]))

    model_visual = Sequential()
    model_visual.add(LSTM(210, input_shape=(1, max_features_v), return_sequences=True))
    model_visual.add(LSTM(120, return_sequences=True))
    #model_visual.add(Dropout({{uniform(0, 1)}}))
    #model_visual.add(Dense(y)

    ## Merge models
    ## - Sequential cannot be used to concatenate, so we have to use the functional API
    out = Concatenate()([model_auditory.output, model_visual.output])

    out = LSTM({{choice([20, 30, 40, 50, 60, 100])}})(out)

    # Avoid overfitting
    #out = Dropout({{uniform(0, 1)}})(concatenated)
    out = Dropout({{uniform(0, 1)}})(out)
    # Regular dense nn with sigmoid activation function
    out = Dense(y_train_a.shape[1], activation='softmax')(out)

    model = Model(inputs = [model_auditory.input, model_visual.input], outputs = out)
 
    ## Compile model
    model.compile(
        loss='categorical_crossentropy'
      , optimizer='rmsprop'
      , metrics = ['accuracy']    # Collect accuracy metric 
    )

    ## Early stop
    early_stopping = EarlyStopping(monitor='loss', patience=8)

    ## Fit model
    model.fit([X_train_a, X_train_v], y_train_a, 
              batch_size=128,
              epochs=500,
              validation_data=([X_test_a, X_test_v], y_test_a),
              callbacks=[early_stopping])

    ## Extract score)
    score, acc = model.evaluate([X_test_a, X_test_v], y_test_a, verbose=0)

    print("Accuracy: ", acc)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model, data=data, algo=tpe.suggest, max_evals=10, trials=Trials())
    #X_train_a, y_train_a, X_test_a, y_test_a, max_features_a, maxlen_a, X_train_v, y_train_v, X_test_v, y_test_v, max_features_v, maxlen_v = data()
    #model(X_train_a, y_train_a, X_test_a, y_test_a, max_features_a, maxlen_a, X_train_v, y_train_v, X_test_v, y_test_v, max_features_v, maxlen_v)
    #X_train, y_train, X_test, y_test, m, n = data()
    #print(best_model.evaluate(X_test, y_test))
    #best_model.evaluate(X_test, y_test)
    print(best_run)
