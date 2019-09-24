#!/bin/bash
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
python image_downloader.py
python preprocess.py
python poketype2.py
