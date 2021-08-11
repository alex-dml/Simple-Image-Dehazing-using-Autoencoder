# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 14:13:36 2021

@author: duminil
"""
import argparse, os
import keras
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
# from keras.callbacks import TensorBoard
from data import VideoFrameGenerator
from model import create_model
from utils import vid_to_frame

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--path_to_csv', default='F:/AE/video/nyu_CSV/*', type=str)
parser.add_argument('--dataset', default= 'F:/nyu_HazyClear/*', type=str)
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--bs', type=int, default=4, help='Batch size')
parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
parser.add_argument('--gpus', type=int, default=2, help='The number of GPUs to use')
parser.add_argument('--gpuids', type=str, default='0', help='IDs of GPUs to use')

args = parser.parse_args()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
train_gen = VideoFrameGenerator(args.path_to_csv, args.dataset,
            batch_size = args.bs,
            nbframe = 8,
            transform=None)

# create model
model = create_model()

optimizer = Adam(lr = args.lr, amsgrad = True)
model.compile(loss = ['mse'], optimizer = optimizer, metrics = ['accuracy'])
model.summary()

train_len = os.listdir(args.path_to_csv.rsplit('*', 1))
model.fit(train_gen, steps_per_epoch = train_len//args.bs, epochs = args.epochs, 
          shuffle = False)

model.save('Z:/AE/video/paths/test_model.h5')

# test video
vid_path = 'Z:/videos_brume/fog.mp4'
dest_folder_path = 'Z:/AE/video/data/frames/'

if os.stat('Z:/AE/video/data/frames/').st_size == 0:
    path_to_frame = vid_to_frame(vid_path, dest_folder_path)
else:
    path_to_frame = dest_folder_path

data = model.predict(path_to_frame)