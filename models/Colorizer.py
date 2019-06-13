from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.utils.vis_utils import plot_model
from keras.models import load_model, save_model

import numpy as np


class Colorizer():
    def __init__(self, model):
        self.model = model
        # train using mean_absolute_error
        self.model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5))

    def summary(self, plot=False, plot_path='model_plot.png'):
        print(self.model.summary())
        if plot:
            plot_model(self.model, to_file=plot_path, show_shapes=True, show_layer_names=True)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path):
        save_model(self.model, path)

    def load(self, path):
        self.model = load_model(path)


class GANColorizer():
    def __init__(self, discriminator, generator):
        self.generator = generator
        self.discriminator = discriminator

        self.build_gan()

    def build_gan(self):

        # combine two network
        self.gan = Sequential()

        self.gan.add(self.generator)
        self.discriminator.trainable = False

        self.gan.add(self.discriminator)

        # compile
        self.generator.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5))
        self.discriminator.trainable = False

        self.gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
        self.discriminator.trainable = True

        self.discriminator.compile(loss='binary_crossentropy', 
            optimizer=Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08))


    def summary(self, plot=False, plot_path='gen_plot.png'):
        print(self.generator.summary())
        if plot:
            plot_model(self.generator, to_file=plot_path, show_shapes=True, show_layer_names=True)

    def predict(self, x):
        return self.generator.predict(x)

    def save(self, path):
        save_model(self.generator, path)

    def load(self, path):
        self.generator = load_model(path)

    def train_op(self, x, y):
        # generate fake samples
        fake_x = self.generator.predict(x)
        real_fake = np.concatenate((y, fake_x))

        # construct labels used in discriminator
        batch_size = x.shape[0]
        real_labels = [1] * batch_size
        fake_labels = [0] * batch_size

        # iteratively train discriminator and chained GAN
        # when training chained GAN, freeze discriminator params

        self.discriminator.trainable = True
        d_loss = self.discriminator.train_on_batch(real_fake, real_labels + fake_labels)
        self.discriminator.trainable = False

        gan_loss = self.gan.train_on_batch(x, real_labels)
        self.discriminator.trainable = True

        return gan_loss, d_loss

    def get_model(self):
        return self.gan
