# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 14:56:55 2021

@author: duminil
"""

from tensorflow.keras import Model
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization, Conv2D, UpSampling2D, concatenate, Dropout, Conv2DTranspose, concatenate
from tensorflow.keras.layers import Input

def build_model():
    
    input_layer = Input(shape=(480, 640, 3))
    
    # Encoder
    conv1 = Conv2D(32,(3,3), activation='relu', padding='same')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32,(3,3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2,2))(conv1)
    pool1 = Dropout(0.25)(pool1)
    
    conv2 = Conv2D(64,(3,3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64,(3,3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2,2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(128,(3,3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128,(3,3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D((2,2))(conv3)
    pool3 = Dropout(0.5)(pool3)
    
    conv4 = Conv2D(256,(3,3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256,(3,3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D((2,2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    # Middle
    convm = Conv2D(512, (3,3), activation='relu', padding='same')(pool4)
    convm = Conv2D(512, (3,3), activation='relu', padding='same')(convm)
    
    # Decoder
    deconv4 = Conv2DTranspose(256,(3,3), strides=(2,2), padding='same')(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(256, (3,3), activation='relu', padding='same')(uconv4)
    uconv4 = BatchNormalization()(uconv4)
    uconv4 = Conv2D(256, (3,3), activation='relu', padding='same')(uconv4)
    uconv4 = BatchNormalization()(uconv4)
    
    deconv3 = Conv2DTranspose(256,(3,3), strides=(2,2), padding='same')(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(128, (3,3), activation='relu', padding='same')(uconv3)
    uconv3 = BatchNormalization()(uconv3)
    uconv3 = Conv2D(128, (3,3), activation='relu', padding='same')(uconv3)
    uconv3 = BatchNormalization()(uconv3)
    
    deconv2 = Conv2DTranspose(256,(3,3), strides=(2,2), padding='same')(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(128, (3,3), activation='relu', padding='same')(uconv2)
    uconv2 = BatchNormalization()(uconv2)
    uconv2 = Conv2D(128, (3,3), activation='relu', padding='same')(uconv2)
    uconv2 = BatchNormalization()(uconv2)
    
    deconv1 = Conv2DTranspose(256,(3,3), strides=(2,2), padding='same')(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(32, (3,3), activation='relu', padding='same')(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Conv2D(32, (3,3), activation='relu', padding='same')(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    
    output_layer = Conv2D(3, (1,1), padding='same', activation='sigmoid')(uconv1)
    
    model = Model(input_layer, output_layer)
    
    return model

