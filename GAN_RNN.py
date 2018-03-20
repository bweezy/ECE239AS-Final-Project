from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Input, LSTM, Dense, Flatten, Reshape, Concatenate, Lambda
from keras import backend
from keras.engine.topology import Container

from read_eeg_file import parse_eeg_data_for_GAN

import numpy as np
import h5py
import matplotlib.pyplot as plt


def EEG_Concatenate(input_list):
  input1, x = input_list

  slice1 = input1[:,:,:9]
  slice2 = input1[:,:,9:]

  slice1 = backend.permute_dimensions(slice1, (0, 2, 1))
  slice2 = backend.permute_dimensions(slice2, (0, 2, 1))
  x = backend.expand_dims(x, axis=1)

  output = Concatenate(axis=1)([slice1, x, slice2])

  output = backend.permute_dimensions(output,(0,2,1))
  return output

class GAN_RNN():

  def __init__(self, gen_input_shape, disc_input_shape):
    self.optimizer = Adam(lr=5e-4, beta_1=.9, beta_2=.99, decay=.99)

    self.generator = self.Generator(input_shape=gen_input_shape)
    self.generator.compile(loss='binary_crossentropy', optimizer=self.optimizer)

    self.discriminator, self.discriminator_fixed = self.Discriminator(input_shape=disc_input_shape)
    self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
    self.discriminator_fixed.trainable = False



    # The generator takes noise as input and generated imgs
    z = Input(shape=gen_input_shape)
    gen_output = self.generator(z)

    valid = self.discriminator_fixed(gen_output)

    # The combined model  (stacked generator and fixed discriminator)
    self.combined = Model(z, valid)
    self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)

  def Generator(self, input_shape):


    input1 = Input(shape=input_shape)
    t, c = input_shape

    intermediate_model = LSTM(100, input_shape=input_shape)(input1)
    intermediate_model_out = Dense(t, activation='tanh')(intermediate_model)

    gen_output = Lambda(EEG_Concatenate, output_shape=(c + 1, t))([input1, intermediate_model_out])

    model = Model(inputs=input1, outputs=gen_output)

    return model

  def Discriminator(self, input_shape):

    model = Sequential()
    model.add(LSTM(100,input_shape=input_shape))
    model.add(Dense(1, activation='sigmoid'))

    inp = Input(shape=input_shape)
    validity = model(inp)

    return Model(inp, validity), Container(inp, validity)


  def train(self, incomplete, complete, epochs, batch_size=128, save_interval=50):


    mini_batch = int(batch_size/2)

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


      gen_eeg = self.generator.predict(inc_eeg)

      if(epoch == 0):
        plt.figure(1)
        plt.subplot(211)
        plt.plot(gen_eeg[0,:,9])
        plt.subplot(212)
        plt.plot(real_eeg[0,:,9])



      results = np.concatenate((np.ones((mini_batch, 1)),np.zeros((mini_batch, 1))))
      eeg_combined = np.concatenate((real_eeg, gen_eeg))
      
      perm = np.random.permutation(eeg_combined.shape[0])
      results = results[perm]
      eeg_combined = eeg_combined[perm]
      

      d_loss = self.discriminator.train_on_batch(eeg_combined, results)

      # ---------------
      # Train Generator
      # ---------------

      idx3 = np.random.randint(0, incomplete.shape[0], mini_batch)
      inc_eeg2 = incomplete[idx3]

      # Generator wants discriminator to think generated eeg are valid
      # Added label smoother
      valid_y = np.array([.9] * mini_batch)

      # Generator Gradient Update
      g_loss = self.combined.train_on_batch(inc_eeg2, valid_y)

      # Plot the progress
      print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))




    plt.figure(2)
    plt.subplot(211)
    plt.plot(gen_eeg[0,:,9])
    plt.subplot(212)
    plt.plot(real_eeg[0,:,9])
    plt.show()


if __name__ == '__main__':

  incomplete, complete = parse_eeg_data_for_GAN(0)


  for i in np.arange(1,9):
    inc, com = parse_eeg_data_for_GAN(i)
    incomplete = np.concatenate((incomplete, inc), axis=0)
    complete = np.concatenate((complete, com), axis=0)


  incomplete = incomplete[:, :, :729]
  complete = complete[:, :, :729]

  incomplete = np.transpose(incomplete, (0, 2, 1))
  complete = np.transpose(complete,(0,2,1))

  gan = GAN_RNN(gen_input_shape=incomplete.shape[1:], disc_input_shape=complete.shape[1:])
  gan.train(incomplete=incomplete, complete=complete, epochs=30, batch_size=128, save_interval=200)


