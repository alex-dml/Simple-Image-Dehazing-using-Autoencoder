import numpy as np
import os
from tensorflow import keras
from sklearn.utils import shuffle
from tensorflow.python.keras.utils.data_utils import Sequence
import cv2

def _resize(img, resolution=480, padding=6):
    from skimage.transform import resize
    return resize(img, (resolution, int(resolution*4/3)), preserve_range=True, mode='reflect', anti_aliasing=True )

    
def get_Image_data(training_path, valid_path, test_path, batch_size, shape_img):
    
    # training data
    hazy_train_dir = os.listdir(os.path.join(training_path,'hazy'))
    hazy_data = [os.path.join(training_path,'hazy',img) for img in hazy_train_dir]
    gt_data = training_path + 'gt/'
    
    train_generator = ImageGenerator2(hazy_train_dir, training_path, batch_size, shape_img, transformation=False)
    
    # validation 
    hazy_val_dir = os.listdir(os.path.join(valid_path,'hazy'))
    hazy_data = [os.path.join(valid_path,'hazy',img) for img in hazy_val_dir]
    gt_data = valid_path + 'gt/'

    valid_generator = ImageGenerator2(hazy_val_dir, valid_path, batch_size, shape_img, transformation=False)
    
    # test 
    hazy_test_dir = os.listdir(os.path.join(test_path,'hazy'))
    hazy_data = [os.path.join(test_path,'hazy',img) for img in hazy_test_dir]
    gt_data = test_path + 'gt/'

    test_generator = ImageGenerator2(hazy_test_dir, test_path, batch_size, shape_img, transformation=False)
    
    return train_generator, valid_generator, test_generator


class ImageGenerator2(Sequence):
    'Generates data for Keras'
    def __init__(self,
                 list_ID_hazy,
                 path_to_data, 
                 batch_size, 
                 shape_img=(256,256,3), 
                 crop_shape=(128,128,3),
                 transformation=False,
                 shuffle = False
                 ):
        'Initialization'
        self.list_ID_hazy = list_ID_hazy
        self.path_to_data = path_to_data
        # self.gt_path = gt_path
        self.batch_size = batch_size
        self.shape_img = shape_img
        self.crop_shape = crop_shape
        self.transformation = transformation
        self.shuffle = shuffle
     
        

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_ID_hazy) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of IDs
        list_IDs_temp = [self.list_ID_hazy[k] for k in indexes]
        # print(list_IDs_temp)
        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_ID_hazy))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        haze, clear = np.zeros( self.shape_img ), np.zeros( self.shape_img )

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
    
            x = cv2.imread(self.path_to_data + 'hazy/' + ID)/255.0
        
            id_ = ID.split('_')[0]

            if not 'SOTS' in self.path_to_data:
                clear_name = os.path.basename(id_) + '.jpg'
            else: 
                clear_name = os.path.basename(id_) + '.png'

            y = cv2.imread(self.path_to_data + 'gt/' + clear_name)/255.0
            
            haze[i] =  _resize(x, 480)
            clear[i] = _resize(y, 480)

        return np.array(haze), np.array(clear)
