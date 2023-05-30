import os  # For interacting with the operating system
import cv2  # For computer vision tasks (for load / process images)
import numpy as np  # For numerical computing and array manipulation 
import matplotlib.pyplot as plt  # For data visualization (for digits)
import tensorflow as tf  # For machine learning 

mnist = tf.keras.datasets.mnist 

# split the data into training data and the test data

# training (image, lable) | test (image, lable)
(x_train, y_train) , (x_test, y_test) = mnist.load_data()

# scale down the data . Normalize the pixel values between 0 and 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# create basic sequential neural network
model = tf.keras.models.Sequential()

# add flatten layer , flatten 2D 28 x 28 into a 1D single layer
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

# add dense layer , a basic fully connected layer . Relu => f(x) = max(0, x)
model.add(tf.keras.layers.Dense(128, activation='relu')) # hidden layer 
model.add(tf.keras.layers.Dense(128, activation='relu')) # hidden layer 
model.add(tf.keras.layers.Dense(10, activation='softmax'))  # output layer => 0,1...9 for digits. Softmask ensures all neurons adds up to 1 (AKA confidence)


# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

