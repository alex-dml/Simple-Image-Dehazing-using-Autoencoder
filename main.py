# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 10:55:00 2021

@author: alxdn
"""
import os, argparse, glob
from model import build_model
from data import get_Image_data
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from metrics import ssim_metric, psnr_metric
# from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.transform import resize

# Argument Parser
parser = argparse.ArgumentParser(description='training')

parser.add_argument('--training_path', default='E:/Datasets/OTS_BETA2/', 
                    type=str, help='Training dataset.')
parser.add_argument('--valid_path', default='E:/Datasets/RESIDE/SOTS/SOTS/outdoor/', 
                    type=str, help='validation dataset.')
parser.add_argument('--test_path', default='E:/Datasets/RESIDE/SOTS/SOTS/indoor/', 
                    type=str, help='test dataset.')

parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--bs', type=int, default=4, help='Batch size')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
args = parser.parse_args()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


shape_img = (args.bs, 480, 640, 3)

train_generator, valid_generator, test_generator = get_Image_data(args.training_path, 
                                                      args.valid_path, 
                                                      args.test_path,
                                                      args.bs, 
                                                      shape_img)

# Create the model
model = build_model()
# model.summary()

my_callbacks = [
    # tf.keras.callbacks.ModelCheckpoint(filepath='/content/drive/My Drive/h5/vid_{epoch:}.h5',
    #                                    monitor="val_loss", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
]

optimizer = Adam(lr=args.lr)
model.compile(optimizer=optimizer, loss = 'mse', metrics=[ssim_metric, psnr_metric])

history = model.fit(train_generator, 
                    validation_data=valid_generator, 
                    epochs=args.epochs, 
                    shuffle=True, 
                    callbacks=my_callbacks)

model.save('test_20_epochs.h5')
# test 

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


image = cv2.imread('E:/Datasets/sots_data/test/hazy/1400.png')/255.0

image = resize(image, (480, int(480*4/3)), preserve_range=True, 
                    mode='reflect', anti_aliasing=True )

image = np.reshape(image,(1, 480, 640, 3))
image = model.predict(image)

cv2.imshow("haze", image[0,:,:,:])
cv2.waitKey()

scores = model.evaluate(test_generator)
print("Scores [loss, ssim, psnr] :", scores)
