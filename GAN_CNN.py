from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras import regularizers, layers
from keras.layers import Input
from keras import backend
from keras.engine.topology import Container


from read_eeg_file import parse_eeg_data_for_GAN

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(1)

def EEG_Concatenate(input_list):

  input1, x = input_list
  
  slice1 = input1[:,:9,:,:]
  slice2 = input1[:,9:,:,:]

  x = backend.expand_dims(x, axis=1)
  x = backend.expand_dims(x, axis=3)

  output = layers.Concatenate(axis=1)([slice1, x])
  output = layers.Concatenate(axis=1)([output,slice2])
  return output



class GAN_CNN():

  def __init__(self, gen_input_shape, disc_input_shape):

    
    self.input_size = gen_input_shape
    self.channels = None

    self.optimizer = Adam(lr=1e-3, beta_1=.9, 
                          beta_2=.99, decay=.99)

    self.generator = self.Generator(input_shape=gen_input_shape)
    
    self.generator.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    self.discriminator, self.discriminator_fixed = self.Discriminator(input_shape=disc_input_shape)
    # For the combined model we will only train the generator
    self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
    self.discriminator_fixed.trainable = False
    
    # The generator takes incomplete data as input and generates 1 channel
    incomplete = layers.Input(shape=gen_input_shape)
    real_complete = layers.Input(shape=disc_input_shape)

    fake_complete = self.generator(incomplete)

    # The valid takes generated eeg data as input and determines validity
    valid = self.discriminator_fixed(fake_complete)

    # The combined model  (stacked generator and discriminator) takes
    # noise as input => generates images => determines validity 
    self.combined = Model([incomplete, real_complete], [valid, fake_complete])
    self.combined.compile(loss=['binary_crossentropy','mse'], loss_weights=[1e-3, 1], optimizer=self.optimizer)
    #self.combined.summary()
    #print self.combined.metrics_names


  def Generator(self, input_shape):
    
    # f.close()
    # Assuming input is 21x729x1
    input1 = Input(shape=(input_shape))
    c,t,d = input_shape

    h = layers.Conv2D(25, kernel_size=(1,11), strides=(1,1), 
                             padding='same', input_shape=input_shape, 
                             kernel_initializer='he_normal',
                             kernel_regularizer=regularizers.l2(.001))(input1)

    h = layers.Conv2D(25, kernel_size=(21,1), strides=(1,1),
                            padding='valid',
                            kernel_initializer='he_normal',
                            kernel_regularizer=regularizers.l2(.001))(h)

    h = layers.LeakyReLU(alpha=0.2)(h)
    h = layers.AveragePooling2D(pool_size=(1,3))(h)


    h = layers.Conv2D(50, kernel_size=(1,11),
                            padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=regularizers.l2(.001))(h)
    h = layers.LeakyReLU(alpha=0.2)(h)
    h = layers.AveragePooling2D(pool_size=(1,3))(h)

    h = layers.Conv2D(100, kernel_size=(1,11), strides=(1,1),
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(.001))(h)
    h = layers.LeakyReLU(alpha=0.2)(h)
    h = layers.AveragePooling2D(pool_size=(1,3))(h)

    h = layers.Conv2D(200, kernel_size=(1,11), strides=(1,1),
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(.001))(h)                          
    h = layers.LeakyReLU(alpha=0.2)(h)
    h = layers.AveragePooling2D(pool_size=(1,3))(h)

    h = layers.Flatten()(h)
    h = layers.Dense(729, activation='tanh')(h)

    gen_output = layers.Lambda(EEG_Concatenate,
                               output_shape=(c+1, t, d))([input1, h])

    model = Model(inputs=input1, outputs=gen_output)

    #model.summary()

    return model





      

  def Discriminator(self, input_shape):


      # Assuming model is not recurrent
      model = Sequential()

      # Architecture Based on EEG Classification Model at https://arxiv.org/pdf/1703.05051.pdf
    
      model.add(layers.Conv2D(25, kernel_size=(1,11), strides=(1,1), 
                               padding='same', input_shape=input_shape, 
                               kernel_initializer='he_normal'))
                               #kernel_regularizer=regularizers.l2(.001)))

      model.add(layers.Conv2D(25, kernel_size=(22,1), strides=(1,1),
                              padding='valid',
                              kernel_initializer='he_normal'))
                              #kernel_regularizer=regularizers.l2(.001)))
      model.add(layers.LeakyReLU(alpha=0.2))
      model.add(layers.AveragePooling2D(pool_size=(1,3)))
      model.add(layers.Dropout(rate=0.2))


      model.add(layers.Conv2D(50, kernel_size=(1,11),
                              padding='same',
                              kernel_initializer='he_normal'))
                              #kernel_regularizer=regularizers.l2(.001)))
      model.add(layers.LeakyReLU(alpha=0.2))
      model.add(layers.AveragePooling2D(pool_size=(1,3)))
      model.add(layers.Dropout(rate=0.2))

      model.add(layers.Conv2D(100, kernel_size=(1,11), strides=(1,1),
                        padding='same',
                        kernel_initializer='he_normal'))
                        #kernel_regularizer=regularizers.l2(.001)))
      model.add(layers.LeakyReLU(alpha=0.2))
      model.add(layers.AveragePooling2D(pool_size=(1,3)))
      model.add(layers.Dropout(rate=0.2))

      model.add(layers.Conv2D(200, kernel_size=(1,11), strides=(1,1),
                        padding='same',
                        kernel_initializer='he_normal'))
                        #kernel_regularizer=regularizers.l2(.001)))                          
      model.add(layers.LeakyReLU(alpha=0.2))
      model.add(layers.AveragePooling2D(pool_size=(1,3)))
      model.add(layers.Dropout(rate=0.2))

      model.add(layers.Flatten())
      model.add(layers.Dense(1, activation='sigmoid'))
      #model.summary()

      eeg = layers.Input(shape=input_shape)
      validity = model(eeg)

      fixed = Container(eeg, validity)

      return Model(eeg, validity), fixed



  def train(self, incomplete, complete, epochs, batch_size=128, save_interval=50):

    # Rescale data set (zero mean, unit variance)

    mini_batch = int(batch_size/2)

    # f = open( 'train_loss.txt', 'w+' )
    f = h5py.File('train_loss.hdf5', 'w')

    f.create_dataset("d_loss", (epochs/save_interval,),  dtype='float64')
    f.create_dataset("acc", (epochs/save_interval,),  dtype='float64')
    f.create_dataset("g_loss", (epochs/save_interval,),  dtype='float64')
	
    global_real_eeg_c10mean = np.mean(complete[:, 9, :729], axis=0)

    for epoch in range(epochs):

      # -------------------
      # Train Discriminator
      # -------------------

	    # random half batch implementaiton?

      # Select a random half batch of real eeg data
      idx1 = np.random.randint(0, complete.shape[0], mini_batch)
      real_eeg = complete[idx1]

      # Select a random half batch of incomplete eeg data
      idx2 = np.random.randint(0, incomplete.shape[0], mini_batch)
      inc_eeg = incomplete[idx1]

	    # Generate fake images (random noise)
      gen_eeg = self.generator.predict(inc_eeg)

      if(epoch == 0):
        plt.figure(1)
        plt.subplot(211)
        plt.plot(gen_eeg[0,9])
        plt.subplot(212)
        plt.plot(real_eeg[0,9])

      real_results = np.concatenate((.9*np.ones((mini_batch, 1)),np.zeros((mini_batch, 1))))
      eeg_combined = np.concatenate((real_eeg, gen_eeg))
      
      perm = np.random.permutation(eeg_combined.shape[0])
      real_results = real_results[perm]
      eeg_combined = eeg_combined[perm]
      

      d_loss = self.discriminator.train_on_batch(eeg_combined, real_results)
	    # ---------------
      # Train Generator
      # ---------------

      idx3 = np.random.randint(0, incomplete.shape[0], batch_size)
      inc_eeg2 = incomplete[idx3]
      real_eeg2 = complete[idx3]


      # Generator wants discriminator to think generated files are valid
      valid_y = np.array([1] * batch_size)
      fake_complete = self.generator.predict(inc_eeg2)

      # Generator Gradient Update
      g_loss = self.combined.train_on_batch([inc_eeg2, fake_complete], [valid_y, real_eeg2])

      # Plot the progress
      #print g_loss

      print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))

      # If at save interval => save generated image samples
      if (epoch % save_interval == 0 or epoch == epochs-1): self.save_imgs(f, epoch, global_real_eeg_c10mean, inc_eeg, d_loss, g_loss, save_interval)


    f.close()
    plt.figure(2)
    plt.subplot(211)
    plt.plot(gen_eeg[0,9])
    plt.subplot(212)
    plt.plot(real_eeg[0,9])
    # plt.show()


  def save_imgs(self, f, epoch, real_avg_eeg, inc_eeg, d_loss, g_loss, save_interval):
    # f = open( 'train_loss.txt', 'a+' )
    # f = h5py.File('train_loss.hdf5', 'a')
    # noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = self.generator.predict(inc_eeg)

    # Rescale images 0 - 1
    # gen_imgs = 0.5 * gen_imgs + 0.5

    #plt.figure(2)
    #plt.subplot(211)
    #plt.plot(gen_imgs[0,9])
    #plt.subplot(212)
    #plt.plot(real_avg_eeg)

    # plt.savefig("gan_checkpoints/train_%d.png" % epoch)
    # f.write("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]\n" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
    f["d_loss"][epoch/save_interval] = d_loss[0] 
    f["acc"][epoch/save_interval] = 100 * d_loss[1]
    f["g_loss"][epoch/save_interval] = g_loss[0] # g_loss is now an array, so need to take first element
    # f['d_loss'].append(d_loss[0])
    # f['acc'].append(100*d_loss[1])
    # f['g_loss'].append(g_loss)
    # f.close()
    plt.close()


if __name__ == '__main__':

  # Load data set

  incomplete, complete = parse_eeg_data_for_GAN(0)


  for i in np.arange(1,9):
    inc, com = parse_eeg_data_for_GAN(i)
    incomplete = np.concatenate((incomplete, inc), axis=0)
    complete = np.concatenate((complete, com), axis=0)




  incomplete = incomplete[:,:,:729, np.newaxis]
  complete = complete[:,:,:729, np.newaxis]

  
  gan = GAN_CNN(gen_input_shape=incomplete.shape[1:], disc_input_shape=complete.shape[1:])
  gan.train(incomplete=incomplete, complete=complete, epochs=50000, batch_size=128, save_interval=100)


  '''
  for i in np.arange(9):
    X, y = parse_eeg_data(0)

    X = X[:,:,:, np.newaxis]
    gan = GAN_CNN(X.shape[1:])

    loss = gan.discriminator.train_on_batch(X, np.ones(X.shape[0]))

    print loss
  '''

