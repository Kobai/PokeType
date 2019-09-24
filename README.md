# PokeType

PokeType: Convolutional Neural Network that predicts a pokemon's primary type based on its appearence. Training and testing data are sprites from generations 1-5.

StatAnalysis: An MLP network that predicts a pokemon's primary type based on its base stats (hp, attack, defense, sp. attack, sp. defense, speed).

Wip: Version 2

To run it on linux/mac (make sure virtualenv is installed)
```
$ chmod a+x run.sh
$ ./run.sh
```
Otherwise
```
$ virtualenv -p python3 venv
$ pip install -r requirements.txt
$ python image_downloader.py
$ python preprocess.py
$ python poketype2.py
```
