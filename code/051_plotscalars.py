#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 18:26:44 2019

@author: luca
"""

import matplotlib
import matplotlib.pyplot as plt
import keras
import numpy as np

from config_script import *


class TrainingPlot(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.losses = []
        self.miou = []
        self.val_losses = []
        self.val_miou = []
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.miou.append(logs.get('mean_iou'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_miou.append(logs.get('val_mean_iou'))

        if len(self.losses) > 1:

            N = np.arange(0, len(self.losses))

            plt.figure()
            plt.plot(N, self.losses, label = "train_loss")
            plt.plot(N, self.val_losses, label = "val_loss")
            plt.title("Training and Validation Loss [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss")
            plt.ylim(top=0.03,bottom=0.01)            
            plt.legend()
            plt.savefig(str(RESULTS_DIRECTORY / 'plots')+'/'+'Loss-Epoch-{}.png'.format(epoch))
            plt.close()
            
            plt.figure()
            plt.plot(N, self.miou, label = "train_mean_iou")
            plt.plot(N, self.val_miou, label = "val_mean_iou")
            plt.title("Training and Validation Mean IOU [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("MeanIOU")
            plt.ylim(top=0.60,bottom=0.45)            
            plt.legend()
            plt.savefig(str(RESULTS_DIRECTORY / 'plots')+'/'+'MeanIOU-Epoch-{}.png'.format(epoch))
            plt.close()        
