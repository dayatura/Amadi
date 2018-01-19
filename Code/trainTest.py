import numpy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#load data
dataset = numpy.load('amadi.npz')
X = dataset['data']
y = dataset['label_code']

#reshape to be [samples][width][height][chanels]

# X = X.reshape(X.shape[0], 28, 28, 1).astype('float32')

# X = X[0:100]
# y = y[0:100]

#normalize the data
X = X / 255.0

# one hot encode output
y = np_utils.to_categorical(y)
num_classes = y.shape[1]

def baseline_model(optimizer='adam'):
	# create model
	model = Sequential()
	model.add(Convolution2D(32, (6, 6), input_shape=(X[1].shape), activation= 'relu' ))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.4))
    model.add(Convolution2D(32, (6, 6), activation= 'relu' ))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# compile model
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

baseline_model().summary()

#### multiple optimizer
# optimizer = ['adam','rmsprop','sgd','adagrad','adadelta','adamax','nadam']
optimizer = ['adam']
model_acc = []
model_loss = []
for op in optimizer:
	model = baseline_model(op)
	history = model.fit(X, y, validation_split=0.33, nb_epoch=100, batch_size=1, verbose=1)
	model_acc.append(history.history['acc'])
	model_loss.append(history.history['loss'])
    model_json = model.to_json()
	model_name = "model_" + op + "_" + str(elapsed_time) + ".json"
	with open(model_name, 'w') as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	weights_name = "weights_" + op + "_" +str(elapsed_time) +  ".h5"
	model.save_weights(weights_name)
	print("Saved model" + weights_name + " to disk")

numpy.savez('result_op_amadi.npz', model_acc=model_acc, model_loss=model_loss, optimizer=optimizer)