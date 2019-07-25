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
Created on Wed May  8 16:00:02 2019

@author: Luca Clissa
"""
from config_script import *
import sys
#tf_path = "/home/luca/Downloads/yes/envs/tensorflow/lib/python3.6/site-packages"
##tf_path = "/home/luca/anaconda3/envs/tf_gpu/lib/python3.7/site-packages/"
#sys.path.append(tf_path)
import cv2
from matplotlib import pyplot as plt
import numpy as np
import random
import Augmentor
import shutil
from tqdm import tqdm

def custom_augmenter(image):
    '''Add salt noise to image. Return: rgb image.'''
    S_P = random.random()
    if (S_P) > 0.2:  # & (not Gauss_noise):
        s_vs_p = 0.02
        amount = 0.004
        #out = image
        # Generate Salt '1' noise
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape[0:2]]
        image[coords] = [255, 255, 255]

    return(image)

augmented_img_path = RAW_DATA_PATH / "train_valid/train_RGB/all_images/images"
augmented_masks_path = RAW_DATA_PATH / "train_valid/train_RGB/all_masks/images"
augmented_masks_path.mkdir(parents=True, exist_ok=True)

def create_pipeline(img_path=IMG_PATH, mask_path=MASKS_PATH,
                    augmented_img_path=augmented_img_path,
                    augmented_masks_path=augmented_masks_path):
    '''Define augmentation pipeline with Augmentor module.
    
    Keyword arguments:
    img_path -- pathlib Path where original images are stored
    mask_path -- pathlib Path where original masks are stored
    augmented_img_path -- pathlib Path where original images are to be saved
    augmented_masks_path -- pathlib Path where original masks are to be saved

    Return: Augmentor Pipeline'''
    # set the augmentation pipeline for training images
    aug_pipeline = Augmentor.Pipeline(
        source_directory=img_path, output_directory=augmented_img_path)
    
    # associate the same pipeline to masks
    aug_pipeline.ground_truth(mask_path)
    
    aug_pipeline.crop_random(
        probability=0.3, percentage_area=0.75, randomise_percentage_area=False)
    aug_pipeline.crop_random(
        probability=0.3, percentage_area=0.61, randomise_percentage_area=False)
    # aug_pipeline.zoom_random(
    #     probability=0.05, percentage_area=0.98, randomise_percentage_area=False)
    
    # rotations
    aug_pipeline.flip_left_right(probability=0.5)
    aug_pipeline.flip_top_bottom(probability=0.5)
    aug_pipeline.rotate_without_crop(probability=0.01, max_left_rotation=10,
                        max_right_rotation=10, expand=True)
    
    # distorsions
    aug_pipeline.random_distortion(probability=0.3, grid_width=8, grid_height=8,
                                   magnitude=2)
    aug_pipeline.gaussian_distortion(probability=0.3, grid_width=4, grid_height=4,
                                     magnitude=8, corner="bell", method="in")
    aug_pipeline.skew(probability=0.1, magnitude=0.1)
    aug_pipeline.shear(probability=0.05, max_shear_left=5, max_shear_right=5)
    
    # contrast/brightness
    aug_pipeline.random_contrast(probability=1, min_factor=0.6, max_factor=1.4)
    aug_pipeline.random_brightness(probability=1, min_factor=0.6, max_factor=1.4)
    
    # erasing
    # aug_pipeline.random_erasing(probability=1, rectangle_area=0.11)
    return(aug_pipeline)    


def augmented_image_dictionary(augmented_img_path=augmented_img_path, tot_img=TOT_IMG):
    '''Navigate augmented results folder and store hash of the augmented images.
    
    Keyword arguments:
    augmented_img_path -- pathlib Path where augmented results are stored
    tot_img -- number of the original images
    
    Return: dictionary with augmented images hash'''
    aug_hash_dict = {}

    counter = 1
    for idx, img_path in tqdm(enumerate(augmented_img_path.iterdir())):
        if 'original' in img_path.name:
            hash_code = img_path.name.split('.TIF_')[1]
            aug_hash_dict[hash_code] = tot_img + counter
            counter += 1
    return(aug_hash_dict)


def split_augmented_images(aug_hash_dict, augmented_img_path=augmented_img_path,
                           image_destination_path=augmented_img_path,
                           mask_destination_path=augmented_img_path):
    '''Navigate augmented results folder and move and rename images and masks.
    
    Keyword arguments:
    aug_hash_dict -- dictionary with hash of the augmented images/masks as coming from augmented_image_dictionary
    augmented_img_path -- pathlib Path where augmented results are stored
    image_destination_path -- pathlib Path where the augmented images are to be saved
    image_destination_path -- pathlib Path where the augmented masks are to be saved
    
    Return: None
    '''
    for _, img_path in tqdm(enumerate(augmented_img_path.iterdir())):
        
        if 'original' in img_path.name:
            #print("\nFilename: ", img_path.name)
            original_name = img_path.name.split('.TIF_')[0].split("original_")[1]
            hash_code = img_path.name.split('.TIF_')[1]
            try:
                idx = aug_hash_dict[hash_code]
            except KeyError:
                print("Error: Hash Code not present in dictionary")
            #print(or_number, idx)
            name_augmented = "aug_campione{}_ORIG_{}{}".format(idx, original_name, ".TIF")
            output_path = image_destination_path / name_augmented
            #print("Augmented image name: ", name_augmented)
            #print("\nOutput path:\n", output_path)
            shutil.move(str(img_path), str(output_path))
    #             break
        elif 'groundtruth' in img_path.name:
            folder = augmented_img_path.name
            #print("\nFilename: ", img_path.name)
            original_name = img_path.name.split('.TIF_')[0].split(folder+"_")[1]
            hash_code = img_path.name.split('.TIF_')[1]
            try:
                idx = aug_hash_dict[hash_code]
            except KeyError:
                print("Error: Hash Code not present in dictionary.\nCheckout mask: {}".format(img_path.name))
            name_augmented = "aug_campione{}_ORIG_{}{}".format(idx, original_name, ".TIF")
            output_path = mask_destination_path / name_augmented
            shutil.move(str(img_path), str(output_path))
        else:
            if img_path.is_file():
                print("\nWARNING: filename not expected.\nCouldn't find {}".format(img_path))
        
    return(None)


def make_augmentation(aug_pipeline, aug_factor, tot_img=TOT_IMG):
    '''Execute augmentation pipeline.
    
    Keyword arguments:
    aug_pipeline -- Augmentor Pipeline with the transformations to apply
    aug_factor -- how many times the original images have to be duplicated
    tot_img -- number of the original images
    
    Return: None
    '''
    aug_pipeline.sample(tot_img * aug_factor)
    return(None)
        

def plot_augmented_images(augmented_img_path=augmented_img_path,
                          augmented_masks_path=augmented_masks_path, 
                          orig_path= RAW_DATA_PATH / "all_images/images/", tot_images=TOT_IMG):
    '''Plot original image with augmentation results.
    
    Keyword arguments:
    augmented_img_path -- pathlib Path where augmented images are stored
    augmented_masks_path -- pathlib Path where augmented masks are stored
    tot_img -- number of the original images
    
    Return: None
    '''
#    image_names_list = [None]*tot_images
    image_names_list = []
    
    for i, img_name in tqdm(enumerate(augmented_img_path.iterdir())):
        if img_name.name.startswith("aug_"):
    #        image_names_list[i] = img_name
            image_names_list.append(img_name)    
        
    augmentation_examples = random.sample(image_names_list,tot_images)
    
    for img_name in augmentation_examples:
        orig_img_name = str(img_name).split('ORIG_')[1]
        orig_img_name = orig_path / orig_img_name
        aug_mask_name = augmented_masks_path / img_name.name
        
        orig_img = cv2.imread(str(orig_img_name), cv2.IMREAD_COLOR)
        aug_img = cv2.imread(str(img_name), cv2.IMREAD_COLOR)
        aug_mask = cv2.imread(str(aug_mask_name), cv2.IMREAD_GRAYSCALE)
    
        plt.figure(figsize=(16, 5))
        plt.subplot(1, 3, 1)
        plt.title("Original image")
        plt.axis('off')
        plt.imshow(orig_img)
        plt.subplot(1, 3, 2)
        plt.title("Augmented image")
        plt.axis('off')
        plt.imshow(aug_img)
        plt.subplot(1, 3, 3)
        plt.title("Augmented mask")
        plt.axis('off')
        plt.imshow(aug_mask, cmap="gray")
        
        plt.suptitle(img_name.name)
        plt.show()
    
    return(None)