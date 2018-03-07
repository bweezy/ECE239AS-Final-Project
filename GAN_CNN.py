from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import regularizers, layers

from read_eeg_file import parse_eeg_data

import numpy as np
import h5py


class GAN_CNN():

  def __init__(self, input_shape):
    self.input_size = None
    self.channels = None

    self.optimizer = Adam(lr=1e-3, beta_1=.9, 
                          beta_2=.99, decay=.99)

    self.generator = self.Generator(output_shape=input_shape)
    self.discriminator = self.Discriminator(input_shape=input_shape)
    self.discriminator.compile(loss='binary_crossentropy', 
            optimizer=self.optimizer,
            metrics=['accuracy'])
    


  def Generator(self, output_shape):
      
    # Look into size of noise vectors effect on generative models
    noise_shape = (100,)
    model = Sequential()
      

  def Discriminator(self, input_shape):


      # Assuming model is not recurrent
      model = Sequential()

      # Architecture Based on EEG Classification Model at https://arxiv.org/pdf/1703.05051.pdf


      # Data = (22,1000,1)
      model.add(layers.Conv2D(25, kernel_size=(1,11), strides=(1,1), 
                               padding='valid', input_shape=input_shape, 
                               kernel_initializer='he_normal',
                               kernel_regularizer=regularizers.l2(.001)))
      # Data = (22,990,25)

      model.add(layers.Conv2D(25, kernel_size=(22,1), strides=(1,1),
                              padding='valid',
                              kernel_initializer='he_normal',
                              kernel_regularizer=regularizers.l2(.001)))
      model.add(layers.LeakyReLU(alpha=0.2))
      
      # Data = (1,990,25)

      model.add(layers.AveragePooling2D(pool_size=(1,3)))

      # Data = (1,330,25)

      # Second Conv Layer
      model.add(layers.Conv2D(50, kernel_size=(1,10),
                              padding='same',
                              kernel_initializer='he_normal',
                              kernel_regularizer=regularizers.l2(.001)))
      model.add(layers.LeakyReLU(alpha=0.2))

      # Data = (1,330,50)

      model.add(layers.AveragePooling2D(pool_size=(1,3)))
      # Data = (1,110,50)

      # Third Conv Layer
      model.add(layers.Conv2D(100, kernel_size=(1,9), strides=(1,1),
                        padding='valid',
                        kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(.001)))
      model.add(layers.LeakyReLU(alpha=0.2))
      model.add(layers.AveragePooling2D(pool_size=(1,3)))
      # Data = (1,34,100)

      # Fourth Conv Layer
      model.add(layers.Conv2D(200, kernel_size=(1,11), strides=(1,1),
                        padding='valid',
                        kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(.001)))                          
      model.add(layers.LeakyReLU(alpha=0.2))
      # Data = (1,24,200)
      model.add(layers.AveragePooling2D(pool_size=(1,3)))
      # Data = (1,8,200)

      model.add(layers.Flatten())
      model.add(layers.Dense(1, activation='sigmoid'))
      #model.summary()

      eeg = layers.Input(shape=input_shape)
      validity = model(eeg)

      return Model(eeg, validity)


if __name__ == '__main__':

  for i in np.arange(9):
    X, y = parse_eeg_data(0)

    X = X[:,:,:, np.newaxis]
    gan = GAN_CNN(X.shape[1:])

    loss = gan.discriminator.train_on_batch(X, np.ones(X.shape[0]))

    print loss


