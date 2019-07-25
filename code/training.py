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
Created on Wed May  8 17:46:18 2019

@author: Luca Clissa
"""

from config_script import *
#from augmentation import custom_augmenter
#import sys
#tf_path = "/home/luca/Downloads/yes/envs/tensorflow/lib/python3.6/site-packages"
#sys.path.append(tf_path)
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model, Model, Sequential
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import Input, Dropout, Activation, Conv2D, MaxPooling2D, UpSampling2D, Lambda, BatchNormalization
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import callbacks, initializers, layers, models, optimizers
from keras import backend as K

import cv2
import random
import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatable


# create path to store validation images separately
valid_img_path = RAW_DATA_PATH / "sample_valid/all_images/images"
valid_masks_path = RAW_DATA_PATH / "sample_valid/all_masks/images"
valid_img_path.mkdir(parents=True, exist_ok=True)
valid_masks_path.mkdir(parents=True, exist_ok=True)

def setup_pipeline(custom_augmentation, training_img_path=SAMPLE_IMG_PATH, 
                   training_masks_path=SAMPLE_MASKS_PATH,
                   valid_img_path=valid_img_path,
                   valid_masks_path=valid_masks_path,
                   val_percentage=0.25, color='rgb', IMG_HEIGHT=512, 
                   IMG_WIDTH=512, BATCH_SIZE=2, VALID_BATCH_SIZE=2, seed=True):
    '''Setup pipeline for images/masks in training and validation.
    
    Keyword arguments:
    custom_augmentation -- custom augmentation function
    training_img_path -- pathlib Path where training images are stored
    training_masks_path -- pathlib Path where training masks are stored
    validation_img_path -- pathlib Path where validation images are stored
    validation_masks_path -- pathlib Path where validation masks are stored
    val_percentage -- percentage of images to set apart as validation set
    color -- color mode of the training/validation images
    IMG_HEIGHT -- height of training images
    IMG_WIDTH -- width of training images
    BATCH_SIZE -- number of images for training batch
    VALID_BATCH_SIZE -- number of images for validation batch
    seed -- seed to use for training/validation split. Random if not specified
    
    Return:
    train_generator -- generator for training images/masks
    validation_generator -- generator for validation images/masks    
    '''
    
    # set seed if not specified. IMPORTANT: seed equal to guarantee that the same transformation is applied to image and mask

    if seed==None:
        seed = random.randint(1, 1000)

    # setup the augmentation chain for images and masks
    image_datagen = ImageDataGenerator(
        rescale=1./255, validation_split=val_percentage, preprocessing_function=custom_augmentation)
    mask_datagen = ImageDataGenerator(
        rescale=1./255, validation_split=val_percentage)
    
    # extract actual parameter for augmentation
    
    # setup training pipeline
    train_image_generator = image_datagen.flow_from_directory(
        str(training_img_path.parent), class_mode=None, seed=seed, subset='training',
        target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, color_mode=color)
    
    train_mask_generator = mask_datagen.flow_from_directory(
        str(training_masks_path.parent), class_mode=None, seed=seed, subset='training',
        target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, color_mode='grayscale')
    
    if valid_img_path:
        # setup validation pipeline
        valid_image_generator = image_datagen.flow_from_directory(
            str(valid_img_path.parent), class_mode=None, seed=seed, subset='validation',
            target_size=(
                IMG_HEIGHT, IMG_WIDTH), batch_size=VALID_BATCH_SIZE, color_mode=color)
    #        save_to_dir=str(valid_img_path), save_prefix='', save_format='TIF')
        
        valid_mask_generator = mask_datagen.flow_from_directory(
            str(valid_masks_path.parent), class_mode=None, seed=seed, subset='validation',
            target_size=(
                IMG_HEIGHT, IMG_WIDTH), batch_size=VALID_BATCH_SIZE, color_mode='grayscale')
    #        save_to_dir=str(valid_masks_path), save_prefix='', save_format='TIF')
    
    train_generator = zip(train_image_generator, train_mask_generator)
    if valid_img_path:
        valid_generator = zip(valid_image_generator, valid_mask_generator)
    else:
        valid_generator = None

    return(train_generator, valid_generator, seed)
    
def make_UNet():
    '''Define simple UNet model. Return: model.'''
    
    inputs = Input((None, None, 3))

    # learn colourspace transformation
    c0 = Conv2D(3, (3, 3), activation='relu',
                kernel_initializer='he_normal', padding='same')(inputs)
    
    c1 = Conv2D(16, (3, 3), activation='relu',
                kernel_initializer='he_normal', padding='same')(c0)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.15)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu',
                kernel_initializer='he_normal', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu',
                kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.25)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu',
                kernel_initializer='he_normal', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    
    u3 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c2)
    u3 = concatenate([u3, c1], axis=3)
    c3 = Conv2D(16, (3, 3), activation='relu',
                kernel_initializer='he_normal', padding='same')(u3)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.15)(c3)
    c3 = Conv2D(16, (3, 3), activation='relu',
                kernel_initializer='he_normal', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c3)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

def res_to_df(results, save=False, history_path=None):
    '''Convert training history to pandas dataframe
        
    Keyword arguments:
    results -- list with output of model.fit_generator
    save -- boolean, whether to save the results
    history_path -- where to store history if save=True (created if not exist)
    
    Return:
    results -- pandas dataframe with training history
    '''
    TOT_EPOCHS = len(results.history["loss"])
    training_history = pd.DataFrame(results.history, index = range(1,TOT_EPOCHS+1))
    if save:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        training_history.to_csv(history_path, sep='\t', index_label="epochs")
    
    return(training_history)

def train(model, train_generator, valid_generator, optimizer, loss,
          custom_metrics, callbacks, epochs, train_steps, valid_steps, 
          save_history, history_path):
    '''Setup pipeline for images/masks in training and validation.
    
    Keyword arguments:
    model -- model architecture to be trained
    train_generator -- generator of training images
    valid_generator-- generator of validation images
    optimizer -- optimization method
    loss -- loss function
    custom_metrics -- list metrics for performance evaluation
    callbacks -- list of callbacks to execute during training
    epochs -- number of training epochs
    train_steps -- number of steps for each training epoch
    valid_steps -- number of steps for each validation epoch
    save_history -- bool for whether to save history df or not
    history_path -- where to save history df when save_history=True
    
    Return:
    model -- model architecture after training
    results -- pandas dataframe of statistics for loss and metrics during training epochs   
    '''
    
    model.compile(optimizer=optimizer, loss=loss, metrics=custom_metrics)
    
    results = model.fit_generator(train_generator, 
                                  steps_per_epoch=train_steps,
                                  validation_data=valid_generator, 
                                  validation_steps=valid_steps,
                                  callbacks=callbacks,
                                  epochs=epochs)
    results_df = res_to_df(results, save=save_history, history_path=history_path)
    
    return(model, results_df)


def plot_metrics_history(history, tot_epochs):
    '''Plot training history for the loss and the custom metrics.
    
    Keyword arguments:
    history -- training history
    tot_epochs -- total number of training epochs
    
    Return: None.
    '''
    plt.figure(figsize=(16,5))
    plt.suptitle("Training history (tot_epochs: 1-{})".format(tot_epochs))
    x_max = tot_epochs + 1 
    # loss
    plt.subplot(1,3,1)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    # plt.yscale('log')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xticks(range(1, x_max, 5))
    plt.legend(['train', 'validation'], loc='upper right')

    # mean iou
    plt.subplot(1,3,2)
    plt.plot(history['mean_iou'])
    plt.plot(history['val_mean_iou'])
    # plt.yscale('log')
    plt.title('model mean iou')
    plt.ylabel('mean iou')
    plt.xlabel('epoch')
    plt.xticks(range(1, x_max, 5))
    plt.legend(['train', 'validation'], loc='upper right')

    # dice coefficient
    plt.subplot(1,3,3)
    plt.plot(history['dice_coef'])
    plt.plot(history['val_dice_coef'])
    # plt.yscale('log')
    plt.title('model dice coefficient')
    plt.ylabel('dice_coef')
    plt.xlabel('epoch')
    plt.xticks(range(1, x_max, 5))
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()
    
    return(None)