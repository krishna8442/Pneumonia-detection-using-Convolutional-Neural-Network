# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 19:55:21 2020

@author: krish
"""

import os
import numpy as np
import pandas as pd 
import random
import matplotlib.pyplot as plt

import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten,MaxPooling2D, Dropout, BatchNormalization
from keras.layers import Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf


train_path='C:/Users/krish/Desktop/coding/chest-xray-pneumonia/chest_xray/train'
validation_path='C:/Users/krish/Desktop/coding/chest-xray-pneumonia/chest_xray/val'
test_path='C:/Users/krish/Desktop/coding/chest-xray-pneumonia/chest_xray/test'

model=Sequential()
model.add(Conv2D(filters=16,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(output_dim = 128, activation = 'relu'))
model.add(Dense(output_dim = 1, activation = 'sigmoid'))


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit_generator(train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800,verbose=1,callbacks=[EarlyStopping(monitor='val_loss',patience=0, verbose=1, mode='min')])

model.evaluate_generator(train_generator, verbose=1)

model.predict_generator(test_generator,verbose=1)





















































