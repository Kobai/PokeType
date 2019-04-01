import glob
import os

import cv2
import matplotlib.pyplot as pyplot
import pandas as pd
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.utils import np_utils
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from sklearn.model_selection import train_test_split

IMG_DIM = (96,96,3)

# Get img directory
files_dir = os.getcwd() + '/imgs/'
print(files_dir);

#  Read all of the images and preprocess them
X =[]
for i in range(649):
    img = cv2.imread(files_dir+str(i+1)+".png")
    img=cv2.resize(img,(96,96))
    img=img_to_array(img)
    X.append(img)

# Scale rgb values so neural net doesn't blow up. Highest intensity is 255 so 255 is a good scaling factor
X = np.array(X, dtype="float")/255.0

# Get the primary type of each pokemon from the csv file. Will even tually add secondary type too
df = pd.read_csv("pokemon_stat.csv")
Y = df[['type_1']].values
Y = Y[:649]
lb = LabelBinarizer()
Y = lb.fit_transform(Y)

# Create training and testing sets
(trainX, testX, trainY, testY) = train_test_split(X,Y,test_size=0.2, random_state=42)

model = Sequential()

model.add(Conv2D(32, (3,3), padding='same', input_shape = (96,96,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3,3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(32,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(32,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(18, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, batch_size=128, epochs=40, verbose=1)
score = model.evaluate(testX, testY, batch_size=128, verbose=1)
print("Test score:", score[0])
print("Test Accuracy", score[1])

model.save("trained_model.h5")
