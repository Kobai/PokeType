import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# Got data from csv
df = pd.read_csv("pokemon_stat.csv")

# feature columns. the highest stat is between 250-300, so 300 is a good scaling factor
X = df.drop('type_1', axis=1).values / 300

# label
Y = df[['type_1']].values
lb = LabelBinarizer()
Y = lb.fit_transform(Y)

# split data into training and testing
(trainX, testX, trainY, testY) = train_test_split(X,Y,test_size=0.2, random_state=42)

# Neural Net
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

#train
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(
   trainX,trainY,
    epochs=300,
    verbose = 2
)

# test
score = model.evaluate(testX, testY, batch_size=128, verbose=1)
print("Test score:", score[0])
print("Accuracy: ", score[1])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)