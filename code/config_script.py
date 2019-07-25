#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Copyright 2018 Luca Clissa, Marco Dalla, Roberto Morelli
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#http://www.apache.org/licenses/LICENSE-2.0
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

"""
Created on Fri May  3 16:43:27 2019

@author: Luca Clissa
"""

### IMPORT LIBRARIES ###

# OS operations
#from keras.preprocessing.image import ImageDataGenerator
#from keras.models import load_model, Model, Sequential
#from keras.layers.pooling import MaxPooling2D
#from keras.layers.merge import concatenate
#from keras.layers.core import Dropout, Lambda
#from keras.layers.convolutional import Conv2D, Conv2DTranspose
#from keras.layers import Input, Dropout, Activation, Conv2D, MaxPooling2D, UpSampling2D, Lambda, BatchNormalization
#from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
#from keras import callbacks, initializers, layers, models, optimizers
#from keras import backend as K
#import tensorflow as tf
import getpass
import os
from pathlib import Path
import sys

# utilities
import datetime
import itertools
from itertools import chain
import random
import re
from tqdm import tqdm

# data handling
import numpy as np
import pandas as pd
#
## image manipulation and visualization
#import cv2
#import imageio
#from matplotlib import pyplot as plt

__all__ = [
        'IMG_CHANNELS',
        'IMG_WIDTH',
        'IMG_HEIGHT',
        'IMG_CHANNELS',
        'RADIUS_MAR',
        'RADIUS_RT',
        'PROJECT_DIRECTORY',
        'sample_images',
        'RAW_DATA_PATH',
        'CODE_DIRECTORY',
        'TEST_IMG_PATH',
        'TEST_MASKS_PATH',
        'IMG_PATH',
        'MASKS_PATH',
        'SAMPLE_IMG_PATH',
        'SAMPLE_MASKS_PATH',
        'count_files_in_directory',
        'TOT_IMG',
        'TEST_IMG',
        'RESULTS_DIRECTORY',
        'MODEL_CHECKPOINTS'
        ]


# ML

### SET IMPORTANT PATHS ###

# raw data path
#RAW_DATA_PATH = Path(
#    "/home/luca/Desktop/Dottorato/Applied Machine Learning/project/raw_data_green")
RAW_DATA_PATH = Path(
    "../raw_data_green")
    
# project and python code paths
PROJECT_DIRECTORY = RAW_DATA_PATH.parent
CODE_DIRECTORY = PROJECT_DIRECTORY / 'code'

# masks path and create directory if does not exist
MASKS_PATH = RAW_DATA_PATH / "all_masks/images"
MASKS_PATH.mkdir(parents=True, exist_ok=True)

# image path and create directory if does not exist
IMG_PATH = RAW_DATA_PATH / "all_images/images"
IMG_PATH.mkdir(parents=True, exist_ok=True)

# masks path and create directory if does not exist
SAMPLE_MASKS_PATH = RAW_DATA_PATH / "sample_masks/images"
SAMPLE_MASKS_PATH.mkdir(parents=True, exist_ok=True)

# image path and create directory if does not exist
SAMPLE_IMG_PATH = RAW_DATA_PATH / "sample_images/images"
SAMPLE_IMG_PATH.mkdir(parents=True, exist_ok=True)


# test image path and create directory if does not exist
TEST_IMG_PATH = RAW_DATA_PATH / "test/all_images/images"
TEST_IMG_PATH.mkdir(parents=True, exist_ok=True)

# masks path and create directory if does not exist
TEST_MASKS_PATH = RAW_DATA_PATH / "test/all_masks/images"
TEST_MASKS_PATH.mkdir(parents=True, exist_ok=True)

# checkpoint path
MODEL_CHECKPOINTS = RAW_DATA_PATH.parent / "results/model_checkpoints"
MODEL_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

RESULTS_DIRECTORY = RAW_DATA_PATH.parent / "results/sample_results"
RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

### SET GLOBAL VARIABLES ###

# set radius for circles representing cells
RADIUS_MAR = 4
RADIUS_RT = 3

# for computing reasons we resize the initial image
IMG_CHANNELS = 3
IMG_WIDTH = 512
IMG_HEIGHT = 512
x_fact = 1600/IMG_WIDTH
y_fact = 1200/IMG_HEIGHT


def count_files_in_directory(count_start=0, dir_list=IMG_PATH):
    '''Count the number of files present in a directory.
    
    Keyword arguments:
    count_start -- initial value for the counting variable
    dir_list -- list of directories to be spanned
    
    Return:
    tot files: count_start plus the number of files in dir_list    
    '''
    
    if type(dir_list) != type([]):
        dir_list = [dir_list]
    tot_files = count_start
    for directory in dir_list:
        tot_files += sum(1 for _ in directory.iterdir() if _.is_file())
    return tot_files


TOT_IMG = count_files_in_directory()
test_percentage = 0.10
TEST_IMG = int(TOT_IMG*test_percentage)

sample_images = ["Mar20bS1C2R2_VLPAGl_200x_g.TIF",
                 "Mar21bS1C1R3_VLPAGr_200x_g.TIF",
                 "RT433S4C1R2_DM_100x_g.TIF",
                 "RT463S3C3R2_MM_100x_g.TIF"]