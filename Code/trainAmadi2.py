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

import time


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


optimizer = ['adam','rmsprop','sgd','adagrad','adadelta','adamax','nadam']
model_acc = []
for op in optimizer:
    def baseline_model(optimizer=op):
        model = Sequential()
        model.add(Convolution2D(32, (6, 6), input_shape=(X[1].shape), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.4))
        model.add(Convolution2D(16, (6, 6), activation= 'relu' ))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=10, batch_size=1, verbose=0)
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
	# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seedt)
    start_time = time.time()
    result = cross_val_score(estimator, X, y, cv=kfold)
    elapsed_time = time.time() - start_time
    model_acc.append([op ,result.mean()*100, result.std()*100, elapsed_time])
    print("Accuracy: %.2f%% (%.2f%%)" % (result.mean()*100, result.std()*100))


numpy.savez('op_amadi2.npz', model_acc=model_acc)