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


IMG_DIM = (96,96,3)

# get img directory
files_dir = os.getcwd() + '\\imgs\\'


X =[]
for i in range(649):
    img = cv2.imread(files_dir+str(i+1)+".png")
    img=cv2.resize(img,(96,96))
    img=img_to_array(img)
    X.append(img)

X = np.array(X, dtype="float")/255.0

df = pd.read_csv("pokemon.csv")
Y = df[['type_1']].values
Y = Y[:649]
lb = LabelBinarizer()
Y = lb.fit_transform(Y)

