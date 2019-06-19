from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.models import load_model, save_model

import numpy as np

ROW = 32
COL = 32
CHANNELS = 1
BaseLevel = ROW//2//2


def AutoEncoder():

    input_img = Input(shape=(ROW,COL,CHANNELS))
    #(32,32,1)
    x = Conv2D(128, (3, 3), activation = 'relu', padding = 'same')(input_img)
    #(32,32,128)
    x = MaxPooling2D((2, 2), padding='same')(x)
    #(16,16,128)
    x = Conv2D(64,(3, 3), activation = 'relu', padding = 'same')(x)
    #(16,16,64)
    x = MaxPooling2D((2, 2), padding='same')(x)
    #(8,8,64)
    x = Flatten()(x)
    #(4096,)

    encoded = Dense(500)(x)
    #(500,)
    oneD = Dense(BaseLevel*BaseLevel*64)(encoded)
    #(4096,)
    fold = Reshape((BaseLevel,BaseLevel,64))(oneD)
    #(8,8,64)

    x = UpSampling2D((2, 2))(fold)
    #(16,16,64)
    x = Conv2D(64,(3, 3), activation = 'relu', padding = 'same')(x)
    #(16,16,64)
    x = UpSampling2D((2, 2))(x)
    #(32,32,64)
    decoded = Conv2D(2, (3, 3), activation='sigmoid', padding='same')(x)
    #(32,32,2)

    autoencoder = Model(input_img, decoded)

    return autoencoder

class AutoEncoderColorizer():
    def __init__(self, model):
        self.model = model
        # train using mean_absolute_error
        self.model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5))

    def summary(self, plot=False, plot_path='Autoencoder_model_plot.png'):
        print(self.model.summary())
        if plot:
            plot_model(self.model, to_file=plot_path, show_shapes=True, show_layer_names=True)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path):
        save_model(self.model, path)

    def load(self, path):
        self.model = load_model(path)

    def get_model(self):
        return self.model