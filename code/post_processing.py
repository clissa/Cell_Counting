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
from math import hypot
from scipy import ndimage
from skimage.filters import sobel
from skimage.morphology import watershed, remove_small_holes, remove_small_objects, label
from skimage.feature import peak_local_max

import cv2
#import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def mask_post_processing(thresh_image, kernel_size=5, min_obj_size=100, max_dist=20):
    '''Post-process a thresholded mask to remove small objects and apply watershed.
    
    Keyword arguments:
    thresh image -- thresholded mask
    kernel_size -- size of the kernel to adopt for opening operation
    min_obj_size -- minimum are accepted, smaller objects are removed
    max_dist -- size parameter for ndimage.maximum_filter function
    
    Return:
    thresh_image -- post-processed mask (uint8)  
    '''
    # post-processing to remove holes and small objects
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel)
    opening = opening.astype("bool")
    thresh_image = remove_small_objects(opening, min_size=min_obj_size, connectivity=1, in_place=False)

    # apply watershed to separate cells better
    distance = ndimage.distance_transform_edt(thresh_image)
    maxi = ndimage.maximum_filter(distance, size=max_dist, mode='constant')
    local_maxi = peak_local_max(maxi, indices=False, footprint=np.ones((3, 3)),
                                labels=thresh_image)
    markers = ndimage.label(local_maxi)[0]
    thresh_image = watershed(-distance, markers,
                             mask=thresh_image, watershed_line=True)
    
    thresh_image = thresh_image.astype("bool")
    thresh_image = remove_small_objects(thresh_image, min_size=20, connectivity=1, in_place=False)

    return(thresh_image.astype("uint8")*255)
    
    
def make_prediction(img_path, threshold, model):
    '''Read an image and return post-processed prediction according to a model.
    
    Keyword arguments:
    img_path -- path of the input image
    threshold -- cutoff for thresholding the raw prediction
    model -- trained model to use for predicting a raw heatmap

    Return:
    thresh_image -- post-processed mask (uint8)  
    '''
    # read input image
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0)

    # compute prediction
    predicted_map = model.predict(img/255.)
    
    # threshold the predicted heatmap
    thresh_image = np.squeeze((predicted_map > threshold).astype('uint8'))
    thresh_image = mask_post_processing(thresh_image)

    return(thresh_image)


def compute_metrics(pred_mask_binary, mask, metrics, img_name):
    '''Read an image and return post-processed prediction according to a model.
    
    Keyword arguments:
    pred_mask_binary -- predicted mask
    mask -- true mask
    metrics -- pandas dataframe to store image-wise metrics
    img_name -- name of the image to be processed

    Return:
    metrics -- pandas dataframe with image-wise metrics
    '''
    # extract predicted objects and counts
    pred_label, pred_count = ndimage.label(pred_mask_binary)
    pred_objs = ndimage.find_objects(pred_label)

    # compute centers of predicted objects
    pred_centers = []
    for ob in pred_objs:
        pred_centers.append(((int((ob[0].stop - ob[0].start)/2)+ob[0].start),
                             (int((ob[1].stop - ob[1].start)/2)+ob[1].start)))

    # extract target objects and counts
    targ_label, targ_count = ndimage.label(mask)
    targ_objs = ndimage.find_objects(targ_label)

    # compute centers of target objects
    targ_center = []
    for ob in targ_objs:
        targ_center.append(((int((ob[0].stop - ob[0].start)/2)+ob[0].start),
                            (int((ob[1].stop - ob[1].start)/2)+ob[1].start)))

    # associate matching objects, true positives
    tp = 0
    fp = 0
    for pred_idx, pred_obj in enumerate(pred_objs):

        min_dist = 30  # 1.5-cells distance is the maximum accepted
        TP_flag = 0

        for targ_idx, targ_obj in enumerate(targ_objs):

            dist = hypot(pred_centers[pred_idx][0]-targ_center[targ_idx][0],
                         pred_centers[pred_idx][1]-targ_center[targ_idx][1])

            if dist < min_dist:

                TP_flag = 1
                min_dist = dist
                index = targ_idx

        if TP_flag == 1:
            tp += 1
            TP_flag = 0

            targ_center.pop(index)
            targ_objs.pop(index)

    # derive false negatives and false positives
    fn = targ_count - tp
    fp = pred_count - tp

    # update metrics dataframe
    metrics.loc[img_name] = [tp, fp, fn, targ_count, pred_count]

    return(metrics)


def F1Score(metrics):
    '''Compute the F1 score based on a dataframe of metrics. Return: list of metrics.'''
    # compute performance measure for the current quantile filter
    tot_tp_test = metrics["TP"].sum()
    tot_fp_test = metrics["FP"].sum()
    tot_fn_test = metrics["FN"].sum()
    tot_abs_diff = abs(metrics["Target_count"] - metrics["Predicted_count"])
    tot_perc_diff = (metrics["Predicted_count"] - metrics["Target_count"])/(metrics["Target_count"]+10**(-6))
    accuracy = (tot_tp_test + 0.001)/(tot_tp_test +
                                      tot_fp_test + tot_fn_test + 0.001)
    precision = (tot_tp_test + 0.001)/(tot_tp_test + tot_fp_test + 0.001)
    recall = (tot_tp_test + 0.001)/(tot_tp_test + tot_fn_test + 0.001)
    F1_score = 2*precision*recall/(precision + recall)
    MAE = tot_abs_diff.mean()
    MPE = tot_perc_diff.mean()

    return(F1_score, MAE, MPE, accuracy, precision, recall)