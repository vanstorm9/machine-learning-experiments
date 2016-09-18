# Simple CNN model for CIFAR-10
import numpy
import matplotlib.pyplot as plt
from time import time

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

from sklearn.metrics import confusion_matrix

import imageDataExtract as dataset

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
#imgPath = 'images/happy/3_0_49.png'
#imgPath = 'images/angry/nhan_0_9.png'
imgPath = 'images/sad/anthony_0_9.png'
num_classes = 3

output = ['Sad','Happy','Angry']


img = dataset.pathToVector(imgPath)


# normalize inputs from 0-255 to 0.0-1.0
img = img.astype('float32')

print img.shape


img = img / 255.0

# Create the model
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(1, 32, 32), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))



#model.load_weights("models/model-0.h5")
model.load_weights("checkpoint/weights-improvement-75-0.1966-bigger.hdf5")

# Compile model

epochs = 25
lrate = 0.01
decay = lrate/epochs

sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])



# Final evaluation of the model
pred = model.predict_classes(img, 1, verbose=0)

print output[pred[0]]
print ''
print ''
print ''

while True:
	print 'Type in another path to make a prediction:'
	path = raw_input()

	img = dataset.pathToVector(path)

	# normalize inputs from 0-255 to 0.0-1.0
	img = img.astype('float32')

	
	# Final evaluation of the model
	pred = model.predict_classes(img, 1, verbose=0)

	print output[pred[0]]
	print ''
	print ''
	print ''

