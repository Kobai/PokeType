import pandas as pd
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import *
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

df = pd.read_csv("pokemon_stat.csv")
X = df.drop('type_1', axis=1).values / 600

Y = df[['type_1']].values
lb = LabelBinarizer()
Y = lb.fit_transform(Y)

(trainX, testX, trainY, testY) = train_test_split(X,Y,test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(32, input_dim=6, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(18, activation='linear'))
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(
    X,Y,
    epochs=300,
    verbose = 2
)

score = model.evaluate(testX, testY, batch_size=128, verbose=1)
print("Test score:", score[0])
print("Accuracy: ", score[1])