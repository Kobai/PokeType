from flask import Flask, render_template, request
from keras.models import load_model
import urllib.request
import cv2
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import os
import pandas as pd
import numpy as np

model = load_model("trained_model.h5")
files_dir = os.getcwd()

app = Flask(__name__ )

@app.route('/')
def main():
    data = {'name': ''}
    return render_template('index.html', data=data)

@app.route('/test',methods=['GET'])
def imgsend():
    urllib.request.urlretrieve(str(request.args['image']),"local-filename.png")
    
    X = []
    img = cv2.imread(files_dir+'\\local-filename.png')
    img = cv2.resize(img,(96,96))
    img=img_to_array(img)
    X.append(img)
    X = np.array(X, dtype='float')/255.0

    df = pd.read_csv("pokemon.csv")
    Y=df[['type_1']].values
    Y= Y[:649]
    lb=LabelBinarizer()
    Y = lb.fit_transform(Y)
    prediction = model.predict(X)
    prediction = lb.inverse_transform(prediction)
    prediction = prediction
    data = {'name': str(prediction)}
    return render_template('index.html', data=data)

if __name__ == "__main__":
    app.run()

