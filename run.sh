#!/bin/bash
pip3 install -r requirements.txt
python3 image_downloader.py
python3 preprocess.py
python3 poketype2.py