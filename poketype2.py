#%%
import os 
import tensorflow as tf
import numpy as np
import urllib.request
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

#%%
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(475, 475,3)))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Conv2D(32, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2,2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='relu'))
model.add(keras.layers.Dense(18, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1/255)
train_gen = train_datagen.flow_from_directory(
  './imgs',
  target_size=(475, 475),
  batch_size=128,
  class_mode='categorical'
)

history = model.fit_generator(
  train_gen,
  steps_per_epoch=8,
  epochs=8,
  verbose=1
)
