from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.models import load_model, save_model

import numpy as np

def unet(input_channel, output_channel, use_bn=True):

    # input image shape (width, height, channel)
    # for unet generator without gan only one channel of LAB's light layer
    inputs = Input((32, 32, input_channel))

    filters = 64  # number of filters in the first layer

    def conv2d(layer_input, filters, f_size=4, bn=True):
        # encoder layer in unet structure
        e = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        e = LeakyReLU(alpha=0.2)(e)
        if bn:
            e = BatchNormalization(momentum=0.8)(e)
        return e

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0, bn=True):
        # decoder layer in unet structure
        d = UpSampling2D(size=2)(layer_input)
        d = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(d)
        if dropout_rate:    # if dropout rate != 0
            d = Dropout(dropout_rate)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        d = Concatenate()([d, skip_input])
        return d

    # (None, 32, 32, 1) -> (None, 16, 16, 64)
    e1 = conv2d(inputs, filters, bn=False) # first encoding layer does not use bn
    
    # (None, 16, 16, 64) -> (None, 8, 8, 128)
    e2 = conv2d(e1, filters*2, bn=use_bn)
    
    # (None, 8, 8, 128) -> (None, 4, 4, 256)
    e3 = conv2d(e2, filters*4, bn=use_bn)
    
    # (None, 4, 4, 256) -> (None, 2, 2, 512)
    e4 = conv2d(e3, filters*8, bn=use_bn)

    # (None, 2, 2, 512) -> (None, 1, 1, 512)  
    e5 = conv2d(e4, filters*8, bn=use_bn)

    # encoder = Model(inputs, e5, name='unet_encoder')
    # print(encoder.summary())

    d1 = deconv2d(e5, e4, filters*8)
    d2 = deconv2d(d1, e3, filters*4)
    d3 = deconv2d(d2, e2, filters*2)
    d4 = deconv2d(d3, e1, filters)

    d7 = UpSampling2D(size=2)(d4)
    output = Conv2D(output_channel, kernel_size=4, strides=1, padding='same', activation='tanh')(d7)

    return Model(inputs, output, name='unet')

class UnetColorizer():
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

    def get_model(self):
        return self.model