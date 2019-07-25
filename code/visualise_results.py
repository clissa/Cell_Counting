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
Created on Sat Jul 20 13:00:41 2019

@author: Luca Clissa
"""

from config_script import *
from post_processing import mask_post_processing

import cv2
#import random
import numpy as np
import pandas as pd
from math import hypot
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

import matplotlib.patches as mpatches
from scipy import ndimage


def plot_predicted_heatmaps(model, test_img_path, test_masks_path):
    '''Plot original image with true objects and the predicted heatmap.
    
    Keyword arguments:
    test_img_path -- path where the images to be plotted are stored
    test_masks_path -- path where the relative masks are stored
   
    Return: None.
    '''
    
    for idx, img_path in enumerate(test_img_path.iterdir()):
    
        if not img_path.name.startswith("aug_"):

            img_rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            mask_path = test_masks_path / img_path.name
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            # build axes labels
            height, width = img_rgb.shape[:2]
            
            # predictions
            img_rgb = np.expand_dims(img_rgb, 0)
            pred_mask_rgb = np.squeeze(model.predict(img_rgb/255.))
        
            # plot predictions
            fig, axes = plt.subplots(1, 2, figsize=(20, 6))
            fig.suptitle(img_path.name)
        
            # original image + true objects
            axes[0].imshow(np.squeeze(img_rgb), cmap=plt.cm.RdBu)
            axes[0].contour(mask, [0.5], linewidths=1.2, colors='w')
            axes[0].set_title('Original image and mask')
            
            # RGB prediction
            im = axes[1].pcolormesh(np.flipud(pred_mask_rgb), cmap='jet')
            divider = make_axes_locatable(axes[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            axes[1].set_title('Predicted heatmap RGB')

def plot_predicted_mask(model, test_img_path, test_masks_path, threshold, post_processing=True):
    '''Plot original image with true objects and the predicted heatmap.
    
    Keyword arguments:
    test_img_path -- path where the images to be plotted are stored
    test_masks_path -- path where the relative masks are stored
    threshold -- cutoff for thresholding predicted heatmap
    
    Return: None.
    '''
    
    for idx, img_path in enumerate(test_img_path.iterdir()):
    
        if not img_path.name.startswith("aug_"):

            img_rgb = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            mask_path = test_masks_path / img_path.name
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            # predictions
            img_rgb = np.expand_dims(img_rgb, 0)
            pred_mask_rgb = np.squeeze(model.predict(img_rgb/255.))
            thresh_image = np.squeeze((pred_mask_rgb > threshold).astype('uint8'))
            
            # apply post-processing
            if post_processing:
                thresh_image = mask_post_processing(thresh_image)
            
            plot_predictions_with_metrics(np.squeeze(img_rgb), img_path.name, 
                                          thresh_image, mask)

    return(None)
    

def plot_predictions_with_metrics(img, img_name, pred_mask, mask):
    '''Plot original image with bounding boxes for TP, FP, and FN.
    
    Keyword arguments:
    img -- array of the original image
    img_name -- name of the image to print
    pred_mask -- array of the predicted mask
    mask -- groundtruth mask
    
    Return: None.
    '''
    
    pred_mask = pred_mask.astype("bool")

    pred_label, pred_rgb = ndimage.label(pred_mask)
    pred_objs = ndimage.find_objects(pred_label)

    # read mask and extract target objects and counts
    true_label, true_count = ndimage.label(mask)
    true_objs = ndimage.find_objects(true_label)
    
    # compute centers of predicted objects
    pred_centers = []
    for ob in pred_objs:
        pred_centers.append(((int((ob[0].stop - ob[0].start)/2)+ob[0].start),
                             (int((ob[1].stop - ob[1].start)/2)+ob[1].start)))

    # compute centers of target objects
    targ_center = []
    for ob in true_objs:
        targ_center.append(((int((ob[0].stop - ob[0].start)/2)+ob[0].start),
                            (int((ob[1].stop - ob[1].start)/2)+ob[1].start)))

    # associate matching objects, true positives
    tp = 0
    tp_objs = []

    for pred_idx, pred_obj in enumerate(pred_objs):

        min_dist = 31  # 1.5-cells distance is the maximum accepted
        TP_flag = 0

        for targ_idx, targ_obj in enumerate(true_objs):

            dist = hypot(pred_centers[pred_idx][0]-targ_center[targ_idx][0],
                         pred_centers[pred_idx][1]-targ_center[targ_idx][1])

            if dist < min_dist:

                TP_flag = 1
                min_dist = dist
                index_targ = targ_idx
                index_pred = pred_idx

        if TP_flag == 1:
            tp += 1
            TP_flag = 0
            
            cv2.rectangle(img,(pred_objs[index_pred][1].start-10,pred_objs[index_pred][0].start-10),
                          (pred_objs[index_pred][1].stop+10,pred_objs[index_pred][0].stop+10),(0,255,0),3)

            tp_objs.append(pred_objs[index_pred])
            targ_center.pop(index_targ)
            true_objs.pop(index_targ)

    # derive false negatives and false positives
    fp = 0
    for pred_obj in pred_objs:
        if pred_obj not in tp_objs:
            cv2.rectangle(img,(pred_obj[1].start-10,pred_obj[0].start-10),
                          (pred_obj[1].stop+10,pred_obj[0].stop+10),(255,0,0),3)
            fp += 1

    fn = 0
    for targ_obj in true_objs:
        cv2.rectangle(img,(targ_obj[1].start-10,targ_obj[0].start-10),
                      (targ_obj[1].stop+10,targ_obj[0].stop+10),(0,0,255),3)
        fn += 1

    # update metrics dataframe
#    test_metrics_rgb.loc[img_name] = [tp, fp, fn, true_count, pred_rgb]

    ae = abs(true_count - pred_rgb)

    # plot
    legend_background_color = 'steelblue'
    line_thickness = 1.5
    plt.figure(figsize=(12, 12))
    plt.suptitle(img_name)

    plt.imshow(img, cmap=plt.cm.RdBu)
    tp_patch = mpatches.Circle((0.1, 0.1), 0.25, facecolor=legend_background_color,
                            edgecolor="green", linewidth=line_thickness)
    fp_patch = mpatches.Circle((0.1, 0.1), 0.25, facecolor=legend_background_color,
                            edgecolor="red", linewidth=line_thickness)
    fn_patch = mpatches.Circle((0.1, 0.1), 0.25, facecolor=legend_background_color,
                            edgecolor="blue", linewidth=line_thickness)
    ae_patch = mpatches.Circle((0.1, 0.1), 0, facecolor=legend_background_color,
                            edgecolor=legend_background_color, linewidth=line_thickness)
    plt.title("Predicted count: {} - True count: {}".format(pred_rgb, true_count))


    legend = plt.legend([tp_patch, fp_patch, fn_patch, ae_patch], 
                        ["True Positive: {}".format(tp), "False Positive: {}".format(fp),
                         "False Negative: {}".format(fn), "Absolute Error: {}".format(ae)], 
                        bbox_to_anchor=(-0.24, 0.55), loc=2)
    frame = legend.get_frame()
    frame.set_color(legend_background_color)
    plt.show()
    return(None)
    
    
def plot_MAE(test_metrics):
    '''Plot mean absolute error distribution based on pandas dataframe. Return None.'''
    
    sns.set_style('whitegrid')
    
    # N.B. the dataframe must contain true and predicted counts in two columns named as follows
    mae_list = list(abs(test_metrics.Target_count - test_metrics.Predicted_count))
    
    fig = plt.figure(figsize=(15,6))
    suptit = plt.suptitle("Absolute Error Distribution")
    
    color = 'blue'
    
    MAX = max(mae_list)
    
    sb = plt.subplot(1,2,1)
    box=plt.boxplot(mae_list,vert=0,patch_artist=True, labels=[""])
    plt.xlabel("Absolute Error")
    plt.ylabel("MAE")
    
    t = plt.text(2, 1.15, 'Mean Abs. Err.: {:.2f}\nMedian Abs. Err.: {:.2f}\nStd. Dev.: {:.2f}'.format(
    np.array(mae_list).mean(), np.median(np.array(mae_list)), np.array(mae_list).std()),
            bbox={'facecolor': color, 'alpha': 0.5, 'pad': 5})
    
    for patch, color in zip(box['boxes'], color):
        patch.set_facecolor(color)
    _ = plt.xticks(range(0,MAX, 5))
    
    sb = plt.subplot(1,2,2)
    
    dens = sns.distplot(np.array(mae_list), bins = 20, color=color, hist=True, norm_hist=False)
    _ = plt.xlim(0,MAX)
    _ = dens.axes.set_xticks(range(0,max(mae_list),5))
    _ = plt.axvline(np.mean(mae_list), 0,1, color="firebrick", label = "Mean Abs. Err.")
    _ = plt.axvline(np.median(mae_list), 0,1, color="goldenrod", label = "Median Abs. Err.")
    
    # Plot formatting
    leg = plt.legend(title="Model")
    xlab = plt.xlabel('Absolute Error')
    ylab = plt.ylabel('Density')
    
    plt.show()
    return(None)
    
    
def plot_MPE(test_metrics):
    '''Plot mean percentage error distribution based on pandas dataframe. Return None.'''
    
    sns.set_style('whitegrid')
    
    # N.B. the dataframe must contain true and predicted counts in two columns named as follows
    mpe_list = list((test_metrics.Predicted_count - test_metrics.Target_count)/(test_metrics.Target_count + 10**(-6)))

    fig = plt.figure(figsize=(15,6))
    suptit = plt.suptitle("Percentage Error Distribution")
    
    color = 'green'
    
    MIN = min(mpe_list)
    MAX = max(mpe_list)
    
    sb = plt.subplot(1,2,1)
    box=plt.boxplot(mpe_list,vert=0,patch_artist=True, labels=[""])
    plt.xlabel("Percentage Error")
    plt.ylabel("MPE")
    
    t = plt.text(-0.9, 1.15, 'Mean Perc. Err.: {:.2f}\nMedian Perc. Err.: {:.2f}\nStd. Dev.: {:.2f}'.format(
    np.array(mpe_list).mean(), np.median(np.array(mpe_list)), np.array(mpe_list).std()),
            bbox={'facecolor': color, 'alpha': 0.5, 'pad': 5})
    
    for patch, color in zip(box['boxes'], color):
        patch.set_facecolor(color)
    # _ = plt.xticks(range(0,MAX, 5))
    
    sb = plt.subplot(1,2,2)
    
    dens = sns.distplot(np.array(mpe_list), bins = 20, color=color, hist=True, norm_hist=False)
    _ = plt.xlim(MIN,MAX)
    # _ = dens.axes.set_xticks(range(0,max(mae_list),5))
    _ = plt.axvline(np.mean(mpe_list), 0,1, color="firebrick", label = "Mean Perc. Err.")
    _ = plt.axvline(np.median(mpe_list), 0,1, color="goldenrod", label = "Median Perc. Err.")
    
    # Plot formatting
    leg = plt.legend(title="Model")
    xlab = plt.xlabel('Percentage Error')
    ylab = plt.ylabel('Density')
    
    plt.show()
    return(None)