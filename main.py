#initial iteration-- Refer jpyter or ann.py

# pip install matplotlib, numpy, tensorflow, open cv(computer vision) python

#installing libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten

#loading the dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#normalize or scling down to values b/w 0 and 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#model building
model = tf.keras.model.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

#compile the model
model.compile(optimize='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#fit/train the model
model.fit(x_train, y_train, epoch=3)

model.save('handwritten.model')