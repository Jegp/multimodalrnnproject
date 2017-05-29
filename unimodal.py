import keras
import numpy as np
import keras as keras
import keras.layers.core
import keras.models 
import keras.optimizers
import keras.utils
import keras.preprocessing.sequence

## Generate test data (20, 53)
data = np.random.random((100, 53, 20))
labels = keras.utils.to_categorical(np.random.randint(32, size=(100, 1)), num_classes=32)

data_test = np.random.random((100, 53, 20))
labels_test = keras.utils.to_categorical(np.random.randint(32, size=(100, 1)), num_classes=32)

# as the first layer in a Sequential model
model = keras.models.Sequential()
model.add(keras.layers.recurrent.LSTM(32, input_shape=(53, 20)))
# Avoid overfitting
model.add(keras.layers.Dropout(0.5))
# Regular dense nn with sigmoid activation function
model.add(keras.layers.Dense(32, activation='sigmoid'))

# now model.output_shape == 1)

## Compile model
model.compile(loss='mean_squared_error', optimizer='sgd')

## Fit model
model.fit(data, labels, epochs=10, batch_size=128)

## Extract score)
score = model.evaluate(data_test, labels_test, batch_size=128)

print("Score: {:f}".format(score))
