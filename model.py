# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 14:07:11 2021

@author: duminil
"""

from tensorflow.keras import Model
from tensorflow.keras.layers import MaxPooling3D, Conv3D, Dropout, Activation, UpSampling3D, concatenate
from tensorflow.keras.layers import Input

def create_model():

    input_img = Input(shape=(8, 480, 640, 3))
    
    ### [First half of the network: downsampling inputs] ###
    kernel_size = (8,8,8)
    
    # Encoder
    conv1 = Conv3D(16, kernel_size, activation = 'relu', padding = 'same')(input_img)
    pool1 = MaxPooling3D((2,2,2), padding = 'same')(conv1)
    
    drop = Dropout(0.5)(pool1)
    
    conv2 = Conv3D(32, kernel_size, activation = 'relu',padding = 'same')(drop)
    encoded = MaxPooling3D((2,2,2), padding = 'same')(conv2)
    
    # Decoder
    conv4 = Conv3D(32, kernel_size, activation = 'relu', padding = 'same')(encoded)
    up4  = UpSampling3D((2,2,2))(conv4)
    # merge4 = concatenate([conv3, up4], axis = 3)
    
    conv5 = Conv3D(16, kernel_size, activation = 'relu', padding = 'same')(up4)
    up5  = UpSampling3D((2,2,2))(conv5)
    # merge5 = concatenate([conv2, up5], axis = 3)

    # merge6 = concatenate([conv1, up6], axis = 3)
    decoded = Conv3D(3, kernel_size, activation = 'relu', padding = 'same')(up5)
    
    model = Model(input_img, decoded)

    return model

def get_model():
    # Define model
    input_img = Input(shape=(3, 480, 640, 3))
    
    x =  Conv3D(32, kernel_size=(3, 3, 3), padding='same')(input_img)
    
    x = Activation('relu')(x)
    x = Conv3D(32, kernel_size=(3, 3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(3, 3, 3), padding='same')(x)
    x = Dropout(0.25)(x)

    x = Conv3D(64, kernel_size=(3, 3, 3), padding='same')(x) 
    x = Activation('relu')(x)
    x = Conv3D(64, kernel_size=(3, 3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling3D(pool_size=(3, 3, 3), padding='same')(x)
    x = Dropout(0.25)(x)

    model = Model(input_img, x)
    

    #plot_model(model, show_shapes=True,
    #           to_file='model.png')
    
    return model