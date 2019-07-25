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
Created on Fri May  3 17:19:59 2019

@author: Luca Clissa
"""
# import libraries
from config_script import *
import seaborn as sns
#import sys
#tf_path = "/home/luca/Downloads/yes/envs/tensorflow/lib/python3.6/site-packages"
#sys.path.append(tf_path)
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
from matplotlib import colors
import numpy as np



# set grid
sns.set(style="whitegrid")

img_names = ["Mar20bS1C2R2_VLPAGl_200x_g.TIF",
                 "Mar21bS1C1R3_VLPAGr_200x_g.TIF",
                 "RT433S4C1R2_DM_100x_g.TIF",
                 "RT463S3C3R2_MM_100x_g.TIF"]

# plot original images
def plot_images(sample_images = img_names):
    '''Plot sample images. Return: None'''
    plt.figure(figsize=(16, 12))
    for i, img_name in enumerate(sample_images):
        img_path = RAW_DATA_PATH / img_name
        # read image in BGR
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        # backtransform to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 2, i+1)
        plt.title(img_name)
        plt.axis('off')
        plt.imshow(img)
    plt.show()
    return(None)
    
# violin plot of pixel intensity
def violin_plot(sample_images = img_names):
    '''Violin plot of the sample images. Return: None'''
    red_channels = []
    green_channels = []
    blue_channels = []
    # plot pixel distribution
    for i, img_name in enumerate(sample_images):
        img_path = RAW_DATA_PATH / img_name
        # read image in BGR
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        # backtransform to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r, g, b = cv2.split(img)
        red_channels.append(r.flatten())
        green_channels.append(g.flatten())
        blue_channels.append(b.flatten())
    
    dfr = pd.DataFrame(list(map(list, zip(*red_channels))),
                       columns=sample_images).assign(Color="Red")
    dfg = pd.DataFrame(list(map(list, zip(*green_channels))),
                       columns=sample_images).assign(Color="Green")
    dfb = pd.DataFrame(list(map(list, zip(*blue_channels))),
                       columns=sample_images).assign(Color="Blue")
    cdf = pd.concat([dfr, dfg, dfb])
    mdf = pd.melt(cdf, id_vars=['Color'], var_name=[
                  'Image'], value_name="Intensity")
    fig = plt.figure(figsize=(15, 5))
    ax = sns.violinplot(x="Color", y="Intensity", palette="Pastel2",
                        hue="Image", data=mdf)
    ax.set_title('Violin plot of pixel intensity divided by channels')
    plt.show()
    return(None)
    
# plot colorspace 3D
def plot_3Dcolorspace(img, colorspace=['RGB', 'HSV'], img_name=None):
    '''Plot pixel into 3D colorspace.
    
    Keyword arguments:
    img -- array with rgb image
    colorspace -- list with names of the colorspace to consider; only rgb and hsv supported
    img_name -- name of the image to use as title of the plot
    
    Return: None
    '''
    # store pixel colors into a list
    pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    # print 3D scatterplot of pixels into the appropriate colorspace

    # RGB
    fig = plt.figure(figsize=(15, 5))
    axis = fig.add_subplot(1, 2, 1, projection="3d")
    r, g, b = cv2.split(img)
    axis.scatter(r.flatten(), g.flatten(), b.flatten(),
                 facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.title("{} colorspace".format(colorspace[0]))

    # HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_img)
    axis = fig.add_subplot(1, 2, 2, projection="3d")
    axis.scatter(h.flatten(), s.flatten(), v.flatten(),
                 facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.title("{} colorspace".format(colorspace[1]))
    if img_name:
        fig.suptitle(img_name)
    plt.show()
    return(None)

#for i, img_name in enumerate(img_names):
#    img_path = RAW_DATA_PATH / img_name
#    # read image in BGR
#    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
#    # backtransform to RGB
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    plot_3Dcolorspace(img, ['RGB', 'HSV'], img_name)
#    plt.show()

if __name__ == "__main__":
    main()