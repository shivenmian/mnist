"""
Handwritten digits recognition in Keras using MLP
"""

from keras.datasets import mnist
from keras.models import Sequential 
from keras.layers import Dense, Activation, advanced_activations
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
np.random.seed(42)

pixelsize = X_train.shape[1] * X_train.shape[2]
X_train = (X_train.reshape(X_train.shape[0], pixelsize).astype('float32'))/255
X_test = (X_test.reshape(X_test.shape[0], pixelsize).astype('float32'))/255
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
mlpmod = Sequential()

lrelu = advanced_activations.LeakyReLU(alpha=0.3)

mlpmod.add(Dense(pixelsize, input_dim=pixelsize, activation=lrelu))
mlpmod.add(Dense(Y_test.shape[1], init='normal', activation='softmax'))
mlpmod.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#hold out validation for simplicity
mlpmod.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=10, batch_size=200, verbose=2)
acc = mlpmod.evaluate(X_test, Y_test, verbose=1)
print("Accuracy: %.2f%%" % (acc[1]*100))