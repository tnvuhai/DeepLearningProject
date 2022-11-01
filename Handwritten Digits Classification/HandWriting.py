# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 07:50:03 2022

@author: Nguyen Vu Hai
"""

#Importing Packages
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
#from tensorflow.python.client import device_lib
#tf.test.gpu_device_name()

#Importing Dataset
from keras.datasets import mnist
(train_img, train_lab), (test_img, test_lab) = mnist.load_data()

#Normalizing Dataset
train_img = train_img.reshape(60000, 28,28,1)
test_img = test_img.reshape(10000,28,28,1)
train_img = keras.utils.normalize(train_img, axis=1)
test_img = keras.utils.normalize(test_img, axis =1)

#Building Model
model=Sequential()
model.add(Conv2D(32,(3,3), input_shape=(28,28,1)))
model.add(MaxPooling2D(3,3))
model.add(Conv2D(16, (3,3)))
model.add(MaxPooling2D(3,3))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation="softmax"))

#Compiling Model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

#Fitting the Model
model.fit(train_img, train_lab, epochs=10)

#Evaluate the Model
Accuracy = model.evaluate(test_img, test_lab)
print(f"Loss: {Accuracy[0]}\nAccuracy: {Accuracy[1]}")

#Predicting First 10 test images
pred = model.predict(test_img[10:20])
# print(pred)
p=np.argmax(pred, axis=1)
print(p)
print(test_lab[10:20])

a = [i for i  in range(10,20)]

#Visualizing prediction
for i,j in zip(a,range(10)):
   plt.imshow(test_img[i].reshape((28,28)), cmap='binary')
   plt.title("Original: {}, Predicted: {}".format(test_lab[i], p[j]))
   plt.axis("Off")
   plt.figure()
   

print(model.summary())