from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.models import load_model, save_model

import numpy as np

def ResnetFusionModel():

	embed_input = Input(shape=(1000,))

	#Encoder
	encoder_input = Input(shape=(32, 32, 1,))
	encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
	encoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output)
	encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
	encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
	encoder_output = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
	encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
	encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
	encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)

	#Fusion
	fusion_output = RepeatVector(4 * 4)(embed_input) 
	fusion_output = Reshape(([4, 4, 1000]))(fusion_output)
	fusion_output = concatenate([encoder_output, fusion_output], axis=3) 
	fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output) 

	#Decoder
	decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
	decoder_output = UpSampling2D((2, 2))(decoder_output)
	decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
	decoder_output = UpSampling2D((2, 2))(decoder_output)
	decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
	decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
	decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
	decoder_output = UpSampling2D((2, 2))(decoder_output)

	model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)

	return model

class ResnetFusionColorizer():
    def __init__(self, model):
        self.model = model
        # train using mean_absolute_error
        self.model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5))

    def summary(self, plot=False, plot_path='resnet_fuion_model_plot.png'):
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