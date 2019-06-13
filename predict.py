import numpy as np

from Dataset import CifarGenerator

import cv2
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import lab2rgb, rgb2lab

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os

from Networks import unet
from Colorizer import Colorizer

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def convert2rgb(x, predicted):
    assert x.shape[0] == predicted.shape[0], 'number of samples unequal!'

    n_sample = predicted.shape[0]
    ret = []
    for cnt in range(n_sample):
        predicted_val = predicted[cnt]  # get predicted value of a single image

        predicted_img = np.zeros((32, 32, 3))
        predicted_img[:,:,0] = x[cnt][:,:,0]      # light channel stay same
        predicted_img[:,:,1:] = predicted_val[0]    # fill the last channels with predicted values

        predicted_img = (predicted_img * [100, 255, 255]) - [0, 128, 128]   # reverted from norm

        rgb_predicted = lab2rgb(predicted_img)  # convert lab to rgb
        rgb_predicted = rgb_predicted * [255, 255, 255]
        rgb_predicted = rgb_predicted.astype(np.uint8)
        ret.append(rgb_predicted)
    return ret

def test_on_batch(test_dir, res_dir=None):
     # gpu config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    KTF.set_session(sess)

    colorizer = Colorizer(model=unet(1, 2))
    colorizer.load('unet_20.model')

    batch_size = 32

    test_generator = CifarGenerator(img_dir=test_dir, batch_size=batch_size, is_training=False)
    
    for test_X, f_names in test_generator:
        predicted = colorizer.predict(test_X)
        # print(predicted.shape)
        rgb_predicted = convert2rgb(test_X, predicted)

        if res_dir is not None:
            for img, f_name in zip(rgb_predicted, f_names):
                cv2.imwrite(os.path.join(res_dir, os.path.basename(f_name)), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # test run one epoch        
        break

if __name__ == '__main__':
    test_on_batch(test_dir=TEST_DIR, res_dir=TEST_RES_DIR)

