import pandas as pd
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import *

df = pd.read_csv("pokemon_stat.csv")
X = df.drop('type_1', axis=1).values / 600
y = df[['type_1']].values
print(y)
Y = np_utils.to_categorical(y,num_classes=18)
print(Y)
model = Sequential()
model.add(Dense(32, input_dim=6, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()
model.compile(loss='mse', optimizer='adam')

model.fit(
    X,Y,
    epochs=50,
    verbose = 2
)


