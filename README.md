# DRFL
dynamical recall focal loss
This repository contains the official classification code of DRFL

you can download mushroom dataset from https://www.kaggle.com/datasets/lizhecheng/mushroom-classification/data
and move it to ./data/test/Mushroom and ./data/train/Mushroom

Requirements to execute code with GPU
numpy
sklearn
pillow
joblib
opencv-python
keras == 2.3.1
tensorflow == 2.1.0 (or tensorflow-gpu == 2.1.0)

then,you can run training code and test code directly
python sample_cls.py for classification task or python sample_seg.py for segmentation task

at last,you can get all results during training at training.log file.
