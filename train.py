from Dataset import CifarGenerator

# Colorizer is a high-level wrapper
# detailed network structure defined in Networks
from models.Colorizer import Colorizer, GANColorizer
from models.Networks import unet, discriminate_network

# use tensorboard writing logs
from keras.callbacks import TensorBoard
from utils.visualize import write_log

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

import numpy as np
import os

from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from utils.CONFIG import *

def train_unet(sess):
    KTF.set_session(sess)

    colorizer = Colorizer(model=unet(1, 2))
    # colorizer.summary(plot=True, plot_path='unet.png')

    n_epoch = 10
    batch_size = 32

    train_generator = CifarGenerator(img_dir=TRAIN_DIR, batch_size=batch_size)
    steps = int(np.ceil(train_generator.size / batch_size))

    history = colorizer.model.fit_generator(
        generator=train_generator,
        steps_per_epoch=steps,
        epochs=n_epoch)

    # save model
    save_path = 'unet.model'

    print("Serializing network to '{}'...".format(save_path))
    colorizer.save(save_path)

def train_gan(sess):
    KTF.set_session(sess)

    # create discriminator with input channel=2 (for both fake image and conditional image)
    # create generator using unet with input_channel=1 and output_channel=2
    colorizer = GANColorizer(discriminator=discriminate_network(input_channel=2), generator=unet(1, 2))
    
    # colorizer.summary(plot=True)

    n_epoch = 10
    batch_size = 32

    tb_callback = TensorBoard(LOG_DIR)
    tb_callback.set_model(colorizer.get_model())    # return gan model

    print('finish setting up TensorBoard callback, logging to %s' % LOG_DIR)

    train_generator = CifarGenerator(img_dir=TRAIN_DIR, batch_size=batch_size)
    for epoch_cnt in range(n_epoch):
        print('Epoch %d: ' % epoch_cnt)

        for batch_cnt, (x, y) in enumerate(tqdm(train_generator)):

            gan_loss, d_loss = colorizer.train_op(x, y)

            total_cnt = epoch_cnt * len(train_generator) + batch_cnt
            write_log(tb_callback, ['gan_loss'], [gan_loss], total_cnt)
            write_log(tb_callback, ['d_loss'], [d_loss], total_cnt)
        
        # display loss for every epoch
        print('loss of gan: %f, loss of discriminator: %f' % (gan_loss, d_loss))

    save_path = 'gan.model'

    print("Serializing network to '{}'...".format(save_path))
    colorizer.save(save_path)
    

if __name__ == '__main__':

    # gpu config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    train_gan(sess)
    # train_unet(sess)
    