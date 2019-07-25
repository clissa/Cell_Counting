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
Created on Tue May  7 10:42:13 2019

@author: Luca Clissa
"""

from config_script import *
import cv2
from matplotlib import pyplot as plt
import numpy as np

img_names = ["Mar20bS1C2R2_VLPAGl_200x_g.TIF",
                 "Mar21bS1C1R3_VLPAGr_200x_g.TIF",
                 "RT433S4C1R2_DM_100x_g.TIF",
                 "RT463S3C3R2_MM_100x_g.TIF"]

def image_cropper(img_path, crop_size=512, save_path = None, diagnostics=False):
    '''Split an image into crops based on crop_size.
    
    Keyword arguments:
    img_path -- pathlib Path of the image to crop
    crop_size -- size of the cropped subimage. Only square crops supported. (Default: 512)
    save_path -- pathlib Path of the folder where to store the crops. (Default: None --> meaning no saving)
    diagnostics -- boolean flag for displaying the whole image in terms of crops. (Default: False)
    
    Return: list with single crops as elements
    '''
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    # backtransform to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # get image height and width
    h, w = img.shape[:2]
    
    # compute number of crops height/width-wise
    n_crop_h = np.ceil(h/crop_size)
    n_crop_w = np.ceil(w/crop_size) 
  
    if n_crop_h==1 and n_crop_w==1:
        print("Warning: the image is smaller than the crop size --> no cropping is needed")
#        return(img)
        
    # define start/end points for crops
    n_central_pixels_h = max(h - 2*crop_size, 0)
    n_central_pixels_w = max(w - 2*crop_size, 0)
    
    if n_crop_h != 2:
        no_overlap_pixels_h = n_central_pixels_h/(n_crop_h -2)
    else:
        no_overlap_pixels_h = n_central_pixels_h+1
    if n_crop_w != 2:
        no_overlap_pixels_w = n_central_pixels_w/(n_crop_w -2)
    else:
        no_overlap_pixels_w = n_central_pixels_h+1
    
    overlap_pixels_h = crop_size - no_overlap_pixels_h
    overlap_pixels_w = crop_size - no_overlap_pixels_w

    startpoints_h = [0]
    startpoints_w = [0]
    
    middle_seq_h = np.arange(crop_size - int(overlap_pixels_h/2),
                             h - crop_size - int(overlap_pixels_h/2),
                             crop_size - overlap_pixels_h).astype(int)
    middle_seq_w = np.arange(crop_size - int(overlap_pixels_w/2),
                             w - crop_size - int(overlap_pixels_w/2),
                             crop_size - overlap_pixels_w).astype(int)
    
    startpoints_h.extend(middle_seq_h)
    startpoints_h.append(h - crop_size)
    startpoints_w.extend(middle_seq_w)
    startpoints_w.append(w - crop_size)

    endpoints_h = [x+crop_size for x in startpoints_h]
    endpoints_w = [x+crop_size for x in startpoints_w]
    
    # prepare canvas for plotting
    if diagnostics:
        plt.figure(figsize=(12,12))
        plt.suptitle(img_path.name)
    
    # retrieve crops
    crops = []
    for c_h in range(int(n_crop_h)):
        for c_w in range(int(n_crop_w)):
            crop = img[startpoints_h[c_h]:endpoints_h[c_h],
                       startpoints_w[c_w]:endpoints_w[c_w]]
            crops.append(crop)
            
            # save crop
            if save_path:
                output_name = img_path.name.split('.TIF')[0] + "_crop_{}{}.TIF".format(c_h + 1, c_w + 1)
                output_path = save_path / output_name
                plt.imsave(str(output_path), crop, format='TIF')
            
            # draw single crop onto subplot
            if diagnostics:
                plt.subplot(n_crop_h,n_crop_w, n_crop_w*c_h + c_w + 1)
                plt.imshow(crop)
                plt.title("Crop [{}, {}]".format(c_h + 1, c_w + 1))
                plt.xticks([0,511],[startpoints_w[c_w], endpoints_w[c_w]])
                plt.yticks([0,511],[startpoints_h[c_h], endpoints_h[c_h]])

    # draw the whole image divided into crops
    if diagnostics:
        plt.show()

    return(crops)

#TRAIN_IMG_PATH = RAW_DATA_PATH / "train/all_images/images"
#for i, img_name in enumerate(["aug_campione1931_ORIG_290.TIF"]):
#    img_path = TRAIN_IMG_PATH / img_name
#    image_cropper(img_path, crop_size=512, save_path = RAW_DATA_PATH, diagnostics=True)  

#for i, img_name in enumerate(img_names):
#    img_path = RAW_DATA_PATH / img_name
#    image_cropper(img_path, crop_size=512, save_path = RAW_DATA_PATH, diagnostics=True)  