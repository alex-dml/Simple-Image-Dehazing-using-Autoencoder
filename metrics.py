# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 15:34:49 2022

@author: duminil
"""
import tensorflow as tf

def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def psnr_metric(y_true, y_pred):
  return tf.image.psnr(y_true, y_pred, 1.0, name=None)