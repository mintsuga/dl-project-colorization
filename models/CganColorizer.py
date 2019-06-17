from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

class CganColorizer():
    def __init__(self):
        
        self.h = 32
        self.w = 32
        self.num_channel = 3
        self.image_shape = (self.h, self.w, self.num_channel)
        
        self.image_A = Input(shape=self.image_shape)
        self.image_B = Input(shape=self.image_shape)
        
        self.build_gan()
    
    def build_gan(self):
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
        
        self.generator = self.build_generator()
        self.fake_A = self.generator(self.image_B)
        self.discriminator.trainable = False
        
        self.valid = self.discriminator([self.fake_A, self.image_B])
        self.combined = Model(inputs=[self.image_A, self.image_B], outputs=[self.valid, self.fake_A])
        self.combined.compile(loss=['mse', 'mae'], loss_weights=[1, 100], optimizer=Adam(0.0002, 0.5))
        
        self.disc_patch = (int(self.h/2**4), int(self.w/2**4), 1)
    
    def build_discriminator(self):
        image_A = Input(shape=self.image_shape)
        image_B = Input(shape=self.image_shape)
        
        # concat axis=channel axis
        combined_images = Concatenate(axis=-1)([image_A, image_B])
        
        # layer 1
        d1 = Conv2D(filters=64, kernel_size=4, strides=2, padding='same')(combined_images)
        d1 = LeakyReLU(alpha=0.2)(d1)
        
        # layer 2
        d2 = Conv2D(filters=128, kernel_size=4, strides=2, padding='same')(d1)
        d2 = LeakyReLU(alpha=0.2)(d2)
        d2 = BatchNormalization(momentum=0.8)(d2)
        
        # layer 3
        d3 = Conv2D(filters=128, kernel_size=4, strides=2, padding='same')(d2)
        d3 = LeakyReLU(alpha=0.2)(d3)
        d3 = BatchNormalization(momentum=0.8)(d3)
        
        # layer 4
        d4 = Conv2D(filters=128, kernel_size=4, strides=2, padding='same')(d3)
        d4 = LeakyReLU(alpha=0.2)(d4)
        d4 = BatchNormalization(momentum=0.8)(d4)
        
        # 1-dim output
        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        
        return Model([image_A, image_B], validity)
    
    def build_generator(self):
        
        d0 = Input(shape=self.image_shape)
        
        # layer 1
        d1 = Conv2D(filters=64, kernel_size=4, strides=2, padding='same')(d0)
        d1 = LeakyReLU(alpha=0.2)(d1)
        
        # layer 2
        d2 = Conv2D(filters=128, kernel_size=4, strides=2, padding='same')(d1)
        d2 = LeakyReLU(alpha=0.2)(d2)
        d2 = BatchNormalization(momentum=0.8)(d2)
        
        # layer 3
        d3 = Conv2D(filters=256, kernel_size=4, strides=2, padding='same')(d2)
        d3 = LeakyReLU(alpha=0.2)(d3)
        d3 = BatchNormalization(momentum=0.8)(d3)
        
        # layer 4
        d4 = Conv2D(filters=512, kernel_size=4, strides=2, padding='same')(d3)
        d4 = LeakyReLU(alpha=0.2)(d4)
        d4 = BatchNormalization(momentum=0.8)(d4)
        
        # layer 5
        d5 = Conv2D(filters=512, kernel_size=4, strides=2, padding='same')(d4)
        d5 = LeakyReLU(alpha=0.2)(d5)
        d5 = BatchNormalization(momentum=0.8)(d5)

        # layer 4
        u4 = UpSampling2D(size=2)(d5)
        u4 = Conv2D(filters=512, kernel_size=4, strides=1, padding='same', activation='relu')(u4)
        u4 = BatchNormalization(momentum=0.8)(u4)
        u4 = Concatenate()([u4, d4])
        
        # layer 3
        u3 = UpSampling2D(size=2)(u4)
        u3 = Conv2D(filters=256, kernel_size=4, strides=1, padding='same', activation='relu')(u3)
        u3 = BatchNormalization(momentum=0.8)(u3)
        u3 = Concatenate()([u3, d3])
        
        # layer 2
        u2 = UpSampling2D(size=2)(u3)
        u2 = Conv2D(filters=128, kernel_size=4, strides=1, padding='same', activation='relu')(u2)
        u2 = BatchNormalization(momentum=0.8)(u2)
        u2 = Concatenate()([u2, d2])
        
        # layer 1
        u1 = UpSampling2D(size=2)(u2)
        u1 = Conv2D(filters=64, kernel_size=4, strides=1, padding='same', activation='relu')(u1)
        u1 = BatchNormalization(momentum=0.8)(u1)
        u1 = Concatenate()([u1, d1])
        
        # layer 0
        u0 = UpSampling2D(size=2)(u1)
        
        # 3-dim output
        u0 = Conv2D(self.num_channel, kernel_size=4, strides=1, padding='same', activation='tanh')(u0)
        
        return Model(d0, u0)
