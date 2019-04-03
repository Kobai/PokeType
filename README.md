# PokeType

PokeType: Convolutional Neural Network that predicts a pokemon's primary type based on its appearence. Training and testing data are sprites from generations 1-5.

StatAnalysis: An MLP network that predicts a pokemon's primary type based on its base stats (hp, attack, defense, sp. attack, sp. defense, speed).

Wip: Version 2

To run it on linux/mac
```
$ chmod a+x run.sh
$ ./run.sh
```
Otherwise
```
$ pip3 install -r requirements.txt
$ python3 image_downloader.py
$ python3 preprocess.py
$ python3 poketype2.py
```