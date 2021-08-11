# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 17:03:51 2021

@author: duminil
"""
from tensorflow import keras
import cv2 as cv
import glob
import numpy as np
import os
import random
import keras_preprocessing
import cv2
import pandas as pd
from tensorflow.python.keras.utils.data_utils import Sequence
from model import get_model, create_model
from tensorflow.keras.optimizers import Adam
from utils import vid_to_frame

class VideoFrameGenerator(Sequence):
    '''
        Video frame generator generates batch of frames from a video directory
            videos/class1/file1.avi
            videos/class1/file2.avi
            videos/class2/file3.avi
    '''
    def __init__(self,
                 from_dir,
                 dataset,
                 batch_size=4,
                 shape=(480, 640, 3),
                 nbframe=8,
                 shuffle=False,
                 transform:keras.preprocessing.image.ImageDataGenerator=None
                ):
        
        self.from_dir = from_dir
        self.dataset = dataset
        self.nbframe = nbframe
        self.batch_size = batch_size
        self.target_shape = shape
        self.shuffle = shuffle
        self.transform = transform
        
        # the list of classes, built in __list_all_files
        self.files = []
        self.dataH = []
        self.dataC = []
        
        # prepare the list
        self.__filecount = 0
        self.__list_all_files()
        
    def __len__(self):
        """ Length of the generator
        Warning: it gives the number of loop to do, not the number of files or
        frames. The result is number_of_video/batch_size. You can use it as
        `step_per_epoch` or `validation_step` for `model.fit_generator` parameters.
        """
        return self.__filecount//self.batch_size
    
    def __getitem__(self, index):
        """ Generator needed method - return a batch of `batch_size` video
        block with `self.nbframe` for each
        """
        T = None
        if self.transform:
            T = self.transform.get_random_transform(self.target_shape[:2])
            
        indexes_x = self.dataH[index*self.batch_size:(index+1)*self.batch_size]
        indexes_y = self.dataC[index*self.batch_size:(index+1)*self.batch_size]
        
        X, Y = [], []
        for vH, vC in zip(indexes_x, indexes_y): 
            
            x = vH
            y = vC
            temp_data_list1 = []
            temp_data_list2 = []
            
            for i, j in zip (x, y):
                # load data in x and y
                try:
                    if T:
                        temp_data_list1.append(self.transform.apply_transform(i, T))
                        temp_data_list2.append(self.transform.apply_transform(j, T))
                    else:
                        temp_data_list1.append(i)
                        temp_data_list2.append(j)
                    
                except Exception as e:
                    print (e)
                    print ('error reading file: ', i) 
                    print ('error reading file: ', j) 
                    
                
            X.append(temp_data_list1)
            Y.append(temp_data_list2)
            
        X = np.array(X)
        Y = np.array(Y)
        
        # print("X shape", X.shape)
        # print("Y shape", Y.shape)

        return X, Y
    
    
    def __list_all_files(self):
        """ List and inject images in memory """
        self.__filecount = len(glob.glob(os.path.join(self.from_dir)))
        
        print(self.__filecount//self.batch_size)
        
        files = glob.glob(self.from_dir)
        
        i = 1
        for file in files:
            print('\rProcessing file %d/%d' % (i, self.__filecount), end='')
            i += 1
            # print(file)
            self.__openframe(file)
                  
        # if self.shuffle:
        #     random.shuffle(self.data)
            
            
    def __openframe(self, file):
        """Append ORIGNALS frames in memory, transformations are made on the fly"""
        framesH = []
        framesC = []
        
        # open csv file
        tmp_df = pd.read_csv(file)
        liste_de_frames = tmp_df.values.tolist()
       
        for i in liste_de_frames :
            
            path_H = i[0]
            path_C = i[1]
            # print(self.dataset.rstrip('*') + path_C.rsplit('nyu_CSV/', 1)[1])
            
            frameC = cv2.imread(self.dataset.rstrip('*') + path_C.rsplit('nyu_CSV/', 1)[1])
            frameH = cv2.imread(self.dataset.rstrip('*') + path_H.rsplit('nyu_CSV/', 1)[1])
            
            frameC = frameC/255
            frameH = frameH/255

            framesH.append(frameH)
            framesC.append(frameC)
            
            # print(framesH)
                
        if len(framesH) != len(framesC):
            print("error number of frameC and frameH not equal") 
        
        step = len(framesC)//self.nbframe
        
        framesC = framesC[::step]
        framesH = framesH[::step]
        
        if len(framesC) >= self.nbframe:
            framesC = framesC[:self.nbframe]
            
        if len(framesH) >= self.nbframe:
            framesH = framesH[:self.nbframe]
        
        # add frames in memory
        if len(framesH) == self.nbframe:
            self.dataH.append(framesH)
        else:
            print('\n/%s has not enough frames ==> %d' % (file, len(framesH)))
            
        # add frames in memory
        if len(framesC) == self.nbframe:
            self.dataC.append(framesC)
        else:
            print('\n/%s has not enought frames ==> %d' % (file, len(framesC)))
        
path1 = 'Z:/AE/video/nyu_CSV/*'
train_gen = VideoFrameGenerator(path1,'E:/nyu_HazyClear/*',
            batch_size=4,
            nbframe=8,
            transform=None)
# keras.preprocessing.image.ImageDataGenerator(rotation_range=5, horizontal_flip=True)

model = create_model()
optimizer = Adam(lr=0.0001, amsgrad=True)
model.compile(loss=['mse'], optimizer=optimizer, metrics=['accuracy'])
model.summary()
model.fit(train_gen, steps_per_epoch = 284//4, epochs=3, shuffle = False)

# test video
vid_path = 'Z:/videos_brume/fog.mp4'
dest_folder_path = 'Z:/AE/video/data/frames/'
path_to_frame = vid_to_frame(vid_path, dest_folder_path)

data = model.predict(path_to_frame)