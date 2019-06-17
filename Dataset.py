import os
from tqdm import tqdm
import scipy
import keras
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import skimage
from skimage.color import rgb2gray
from skimage.color import lab2rgb, rgb2lab

class CifarGenerator(keras.utils.Sequence):
    """
    a generator of Cifar dataset
    use 'for' iteration to generate image in batch-manner
    """
    def __init__(self, img_dir, batch_size, color_space, is_training=True, shuffle=True):
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._is_training = is_training

        self.color_space = color_space

        self._img_names = []

        self.indexes = np.arange(len(self.img_names))

        self.w = 32
        self.h = 32

    def __len__(self):
        # how many batches 
        return int(np.ceil(len(self.indexes) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.is_training and self.shuffle:
            np.random.shuffle(self.indexes)

    def _get_item_lab(self, index):
        upper_bound = min(self.size, (index + 1) * self.batch_size)
        indexes = self.indexes[index * self.batch_size:upper_bound]

        X = []
        Y = []
        for index in indexes:
            f_name = self._img_names[index]
            img = img_to_array(load_img(f_name, target_size=(self.w, self.h))) / 255

            # convert rgb image to lab
            lab_image = rgb2lab(img)

            # color values in the different lab layers are in various range
            # L: [0, 100]
            # a&b: [-128, 128]
            lab_image_norm = (lab_image + [0, 128, 128]) / [100, 255, 255]

            X.append(lab_image_norm[:, :, 0])
            Y.append(lab_image_norm[:, :, 1:])     

        assert len(X) == len(Y)

        X = np.asarray(X, dtype=np.float32)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        Y = np.asarray(Y, dtype=np.float32)

        if self.is_training:
            return X, Y
        else:
            # infer process return X and filenames
            return X, [self._img_names[index] for index in indexes]

    def _get_item_rgb(self, index):
        upper_bound = min(self.size, (index + 1) * self.batch_size)
        indexes = self.indexes[index * self.batch_size:upper_bound]

        images_A = []
        images_B = []
        
        for index in indexes:
            f_name = self._img_names[index]
            image_A = scipy.misc.imread(f_name, mode='RGB').astype(np.float)
            image_B = rgb2gray(image_A)
        
            image_A = scipy.misc.imresize(image_A, (self.h, self.w))
            image_B = scipy.misc.imresize(image_B, (self.h, self.w))
            
            # convert gray-scale image to a 3-channel image to fit input shape
            image_B = np.stack((image_B,)*3, axis=-1)
                
            images_A.append(image_A)
            images_B.append(image_B)
        
        # normalization is to bring the values in range [-1.0,1.0]
        images_A = np.array(images_A)/127.5 - 1.
        images_B = np.array(images_B)/127.5 - 1.
        
        
        return images_A, images_B

    def __getitem__(self, index):
        if self.color_space == 'LAB':
            return self._get_item_lab(index)
        else:
            return self._get_item_rgb(index)


    def _load(self):
        for f_name in tqdm(os.listdir(self.img_dir)):
            if os.path.splitext(f_name)[-1] == '.png':
                self._img_names.append(os.path.join(self.img_dir, f_name))

    @property
    def img_names(self):
        if len(self._img_names) == 0:
            self._load()
        return self._img_names

    @property
    def size(self):
        return len(self._img_names)

    @property
    def is_training(self):
        return self._is_training
    

