import keras
import numpy as np
import keras as keras
import keras.layers.core
import keras.layers.merge
import keras.models 
import keras.optimizers
import keras.utils
import keras.preprocessing.sequence

## Generate test data
data_auditory = np.random.random((100, 53, 20))
data_visual = np.random.random((100, 53, 20))

labels = keras.utils.to_categorical(np.random.randint(32, size=(100, 1)), num_classes=32)
 
data_auditory_test = np.random.random((100, 53, 20))
data_visual_test = np.random.random((100, 53, 20))
labels_test = keras.utils.to_categorical(np.random.randint(32, size=(100, 1)), num_classes=32)

## Build auditory model
model_auditory = keras.models.Sequential()
model_auditory.add(keras.layers.recurrent.LSTM(32, input_shape=(53, 20)))

## Build visual model
model_visual = keras.models.Sequential()
model_visual.add(keras.layers.recurrent.LSTM(32, input_shape=(53, 20)))

## Merge models
## - Sequential cannot be used to concatenate, so we have to use the functional API
concatenated = keras.layers.Concatenate()([model_auditory.output, model_visual.output])
# Avoid overfitting
out = keras.layers.Dropout(0.5)(concatenated)
# Regular dense nn with sigmoid activation function
out = keras.layers.Dense(32, activation='sigmoid')(out)

model = keras.models.Model(inputs = [model_auditory.input, model_visual.input], outputs = out)

## Compile model
model.compile(loss='mean_squared_error', optimizer='sgd')

## Fit model
## - Now we have to use two inputs
model.fit([data_auditory, data_visual], labels, epochs=10, batch_size=128)

## Extract score
## - Now we have to use two inputs
score = model.evaluate([data_auditory_test, data_visual_test], labels_test, batch_size=128)

print("Score: {:f}".format(score))
