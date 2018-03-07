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
    self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    self.discriminator = self.Discriminator(input_shape=input_shape)
    self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
    


  def Generator(self, output_shape):
      
    # Look into size of noise vectors effect on generative models
    noise_shape = (100,)
    model = Sequential()


    model.add(layers.Dense(1*9*200, input_shape=noise_shape,
                            kernel_initializer='he_normal',
                            kernel_regularizer=regularizers.l2(.001)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((1,9,200)))
    model.add(layers.UpSampling2D(size=(1,3)))

    model.add(layers.Conv2D(100, kernel_size=(1,11), strides=(1,1),
                            padding='same', 
                            kernel_initializer='he_normal',
                            kernel_regularizer=regularizers.l2(.001)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.UpSampling2D(size=(1,3)))
    model.add(layers.Conv2D(50, kernel_size=(1,11), strides=(1,1),
                            padding='same', 
                            kernel_initializer='he_normal',
                            kernel_regularizer=regularizers.l2(.001)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.UpSampling2D(size=(1,3)))
    model.add(layers.Conv2D(25, kernel_size=(1,11), strides=(1,1),
                            padding='same', 
                            kernel_initializer='he_normal',
                            kernel_regularizer=regularizers.l2(.001)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.UpSampling2D(size=(22,3)))
    model.add(layers.Conv2D(25, kernel_size=(22,1),
                            padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=regularizers.l2(.001)))
    model.add(layers.Conv2D(1, kernel_size=(1,11), strides=(1,1), 
                            padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=regularizers.l2(.001)))

    model.summary()





      

  def Discriminator(self, input_shape):


      # Assuming model is not recurrent
      model = Sequential()

      # Architecture Based on EEG Classification Model at https://arxiv.org/pdf/1703.05051.pdf


      # Data = (22,1000,1)
      model.add(layers.Conv2D(25, kernel_size=(1,11), strides=(1,1), 
                               padding='same', input_shape=input_shape, 
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
      model.add(layers.Conv2D(50, kernel_size=(1,11),
                              padding='same',
                              kernel_initializer='he_normal',
                              kernel_regularizer=regularizers.l2(.001)))
      model.add(layers.LeakyReLU(alpha=0.2))

      # Data = (1,330,50)

      model.add(layers.AveragePooling2D(pool_size=(1,3)))
      # Data = (1,110,50)

      # Third Conv Layer
      model.add(layers.Conv2D(100, kernel_size=(1,11), strides=(1,1),
                        padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(.001)))
      model.add(layers.LeakyReLU(alpha=0.2))
      model.add(layers.AveragePooling2D(pool_size=(1,3)))
      # Data = (1,34,100)

      # Fourth Conv Layer
      model.add(layers.Conv2D(200, kernel_size=(1,11), strides=(1,1),
                        padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(.001)))                          
      model.add(layers.LeakyReLU(alpha=0.2))
      # Data = (1,24,200)
      model.add(layers.AveragePooling2D(pool_size=(1,3)))
      # Data = (1,8,200)

      model.add(layers.Flatten())
      model.add(layers.Dense(1, activation='sigmoid'))
      model.summary()

      eeg = layers.Input(shape=input_shape)
      validity = model(eeg)

      return Model(eeg, validity)



  def train(self, X_train, epochs, batch_size=128, save_interval=50):

    # Rescale data set (zero mean, unit variance)

    small_batch = int(batch_size / 2)

    for epoch in range(epochs):

      # -------------------
      # Train Discriminator
      # -------------------

		# random half batch implementaiton?

        # Select a random half batch of images
        idx = np.random.randint(0, X_train.shape[0], small_batch)
        imgs = X_train[idx]
        noise = np.random.normal(0, 1, (small_batch, 100))

		# Generate fake images (random noise)
        fake_imgs = self.generator.predict(noise)

        # Train
        d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((small_batch, 1)))
        d_loss_fake = self.discriminator.train_on_batch(fake_imgs, np.zeros((small_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

		# ---------------
        # Train Generator
        # ---------------

        noise = np.random.normal(0, 1, (batch_size, 100))

        # Generator wants discriminator to think generated files are valid
        valid_y = np.array([1] * batch_size)

        # Generator Gradient Update
        g_loss = self.combined.train_on_batch(noise, valid_y)

        # Plot the progress
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        # If at save interval => save generated image samples
        # if (epoch % save_interval == 0) self.save_imgs(epoch)





if __name__ == '__main__':

  # Load data set
  X, y = parse_eeg_data(0)

  X = X[:,:,:729, np.newaxis]

  print X.shape
  gan = GAN_CNN(X.shape[1:])
  gan.train(X_train=X, epochs=3, batch_size=32, save_interval=200)


  '''
  for i in np.arange(9):
    X, y = parse_eeg_data(0)

    X = X[:,:,:, np.newaxis]
    gan = GAN_CNN(X.shape[1:])

    loss = gan.discriminator.train_on_batch(X, np.ones(X.shape[0]))

    print loss
  '''

