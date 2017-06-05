import keras
import numpy as np
import matplotlib.pyplot as plt
import keras as keras
import keras.layers.core
import keras.models 
import keras.optimizers
import keras.utils
import keras.preprocessing.sequence

from sklearn.externals import joblib
from keras.utils.np_utils import to_categorical                                                                        

import sys

# Load system arguments as data entry
data_source = sys.argv[1]
data_type = ""
if (sys.argv[2] == 'video'):
	data_type = 0
else:
	data_type = 1

epochs = 100

# Load data
X_train, X_test, y_train, y_test = joblib.load(data_source)[data_type]

# Reshape to fit LSTM requirements for 3d input
data = np.array(X_train)
data = data.reshape((data.shape[0], 1, data.shape[1]))
labels = to_categorical(y_train)

lstm1 = 300
lstm2 = 400
lstm3 = 400
#lstm4 = 400

data_test = np.array(X_test)
data_test = data_test.reshape((data_test.shape[0], 1, data_test.shape[1]))
labels_test = to_categorical(y_test)

print("Training {} data from {} over {} epochs".format(sys.argv[2], data_source, epochs))
print(data.shape)
print(labels.shape)

# as the first layer in a Sequential model
model = keras.models.Sequential()
model.add(keras.layers.recurrent.LSTM(lstm1, 
	  input_shape = (1, data.shape[2]), 
	  return_sequences = True))
model.add(keras.layers.recurrent.LSTM(lstm2,
	  return_sequences = True))
model.add(keras.layers.recurrent.LSTM(lstm3,
	  return_sequences = False))
#model.add(keras.layers.recurrent.LSTM(lstm4,
#	  return_sequences=False))
# Avoid overfitting
model.add(keras.layers.Dropout(0.5))
# Regular dense nn with sigmoid activation function
model.add(keras.layers.Dense(labels.shape[1], activation = 'softmax'))

# now model.output_shape == 1)

## Compile model
model.compile(
    loss='categorical_crossentropy'
  , optimizer='rmsprop'
  , metrics = ['accuracy', 'mae']    # Collect accuracy metric	
)
## Fit model
#data_reshaped = data
history = model.fit(data, labels, epochs=epochs)

# summarize history for accuracy
print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['mean_absolute_error'])
plt.title('Model precision')
plt.ylabel('Precision')
plt.xlabel('Epoch')
plt.legend(['Accuracy', 'Loss', 'MAE'], loc='upper left')
plt.savefig('model_{}_{}.png'.format(sys.argv[2], epochs))

## Extract score)
score = model.evaluate(data_test, labels_test)

print("Score: {}".format(score))
