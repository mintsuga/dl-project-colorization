import numpy as np

from Dataset import CifarGenerator
import cv2

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os

from models.Colorizer import Colorizer, GANColorizer
from models.Networks import unet, discriminate_network

from utils.visualize import convert2rgb
from utils.CONFIG import *

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def test_on_batch(test_dir, res_dir=None):

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

def test_gan(test_dir, res_dir=None):
    colorizer = GANColorizer(discriminator=discriminate_network(input_channel=2), generator=unet(1, 2))
    colorizer.load('gan.model')

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
    # gpu config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    KTF.set_session(sess)
    test_gan('./data/test', './data/res')
    # test_on_batch(test_dir=TEST_DIR, res_dir=TEST_RES_DIR)

