from Dataset import CifarGenerator

from models.CganColorizer import CganColorizer
from models.UnetColorizer import UnetColorizer, unet

from keras.callbacks import TensorBoard
from utils.visualize import write_log

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

import numpy as np
import pickle
import os

from PIL import Image
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from utils.CONFIG import *

SAVED_DIR = './cgan_samples'

def train_unet(sess, use_logs=True):
    print('Start Training UNET model ...')
    KTF.set_session(sess)

    colorizer = UnetColorizer(model=unet(1, 2))

    n_epoch = 20
    batch_size = 64

    train_generator = CifarGenerator(img_dir=TRAIN_DIR, batch_size=batch_size, color_space='LAB')
    steps = int(np.ceil(train_generator.size / batch_size))

    tb_callback = TensorBoard(LOG_DIR)
    tb_callback.set_model(colorizer.get_model())


    history = colorizer.model.fit_generator(
        generator=train_generator,
        steps_per_epoch=steps,
        epochs=n_epoch,
        callbacks=[tb_callback],)

    # save model
    save_path = 'unet_%d.model' % n_epoch

    print("Serializing network to '{}'...".format(save_path))
    colorizer.save(save_path)

def save_intermediate_images(batch_i, images_A, images_B, fake_images_A):
    assert images_A.shape[0] == images_B.shape[0] == fake_images_A.shape[0]
    
    batch_size = images_A.shape[0]
    generated_image = Image.new('RGB', (32*3, 32*batch_size))
    for batch_cnt in range(batch_size):
        image_A = np.uint8((np.array(images_A[batch_cnt]) * 0.5 + 0.5) * 255)
        image_B = np.uint8((np.array(images_B[batch_cnt]) * 0.5 + 0.5) * 255)
        image_fake_A = np.uint8((np.array(fake_images_A[batch_cnt]) * 0.5 + 0.5) * 255)
        
        image_A = Image.fromarray(image_A)
        image_B = Image.fromarray(image_B)
        image_fake_A = Image.fromarray(image_fake_A)
        
        generated_image.paste(image_B,      (0, batch_cnt*32, 32, (batch_cnt+1)*32))
        generated_image.paste(image_fake_A, (32, batch_cnt*32, 32*2, (batch_cnt+1)*32))
        generated_image.paste(image_A,      (32*2, batch_cnt*32, 32*3, (batch_cnt+1)*32))
    
    generated_image.save(SAVED_DIR + "/cgan_%d.jpg" % batch_i, quality=95)

def train_gan(sess, use_logs=True):

    print('Start Training GAN model ...')
    KTF.set_session(sess)

    colorizer = CganColorizer()
    
    n_epoch = 50
    batch_size = 64

    if use_logs:
        # writer = tf.summary.FileWriter('./logs/gan_loss') 
        writer_1 = tf.summary.FileWriter("./logs/g_loss")
        writer_2 = tf.summary.FileWriter("./logs/d_loss")
         
        loss_var = tf.Variable(0.0)
        tf.summary.scalar("loss", loss_var)
         
        write_op = tf.summary.merge_all()
        print('finish setting up TensorBoard logs')

    train_generator = CifarGenerator(img_dir=TRAIN_DIR, batch_size=batch_size, color_space='RGB')
    test_generator = CifarGenerator(img_dir=TEST_DIR, batch_size=16, color_space='RGB')
    
    for epoch_cnt in range(n_epoch):
        print('Epoch %d: ' % epoch_cnt)

        valid = np.ones((batch_size,) + colorizer.disc_patch)
        fake = np.zeros((batch_size,) + colorizer.disc_patch)

        for batch_cnt, (images_A, images_B) in enumerate(tqdm(train_generator)):
            total_cnt = epoch_cnt * len(train_generator) + batch_cnt

            if images_A.shape[0] != batch_size:
                valid = np.ones((images_A.shape[0],) + colorizer.disc_patch)
                fake = np.zeros((images_A.shape[0],) + colorizer.disc_patch)
            
            fake_A = colorizer.generator.predict(images_B)
    
            d_loss_real = colorizer.discriminator.train_on_batch([images_A, images_B], valid)
            d_loss_fake = colorizer.discriminator.train_on_batch([fake_A, images_B], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            g_loss = colorizer.combined.train_on_batch([images_A, images_B], [valid, images_A])

            if use_logs:
                summary = sess.run(write_op, {loss_var: g_loss[0]})
                writer_1.add_summary(summary, total_cnt)
                writer_1.flush()
             
                # for writer 2
                summary = sess.run(write_op, {loss_var: d_loss[0]})
                writer_2.add_summary(summary, total_cnt)
                writer_2.flush()

                # summary = tf.Summary(value=[
                #         tf.Summary.Value(tag='d_loss', simple_value=d_loss[0]), 
                #         tf.Summary.Value(tag='g_loss', simple_value=g_loss[0]),
                #         ]) 
                # writer.add_summary(summary, total_cnt) 

            # record loss every 200 steps
            if total_cnt and not total_cnt % 200:
                print ("[Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f]" % 
                               (total_cnt+1, (epoch_cnt+1)*len(train_generator), 
                                d_loss[0], 100*d_loss[1], g_loss[0]))
                if not total_cnt % 1000:
                    test_A, test_B = test_generator[0]
                    fake_test_A = colorizer.generator.predict(test_B)
                    # save intermediate samples to see gan effects
                    save_intermediate_images(total_cnt, test_A, test_B, fake_test_A)
            
        # display loss for every epoch
        print('End of epoch [%d]. loss of gan: %f, loss of discriminator: %f' % (epoch_cnt, g_loss[0], d_loss[0]))

    colorizer.generator.save_weights('g_weights_%d.h5' % n_epoch)
    colorizer.discriminator.save_weights('d_weights_%d.h5' % n_epoch)

if __name__ == '__main__':

    # gpu config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    train_gan(sess)
    # train_unet(sess)
    