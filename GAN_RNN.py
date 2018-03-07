from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Input, LSTM, Dense, Flatten, Reshape

from read_eeg_file import parse_eeg_data

import numpy as np
import h5py


class GAN_RNN():

	def __init__(self, input_shape):
		self.optimizer = Adam(lr=1e-3, beta_1=.9, beta_2=.99, decay=.99)

		self.generator = self.Generator(output_shape=input_shape)
		# self.generator.compile(loss='binary_crossentropy', optimizer=self.optimizer)

		self.discriminator = self.Discriminator(input_shape=input_shape)
		self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

		# The generator takes noise as input and generated imgs
		z = Input(shape=(1000,100,))
		output = self.generator(z)

		# For the combined model we will only train the generator
		self.discriminator.trainable = False

		# The valid takes generated images as input and determines validity
		valid = self.discriminator(output)

		# The combined model  (stacked generator and discriminator) takes
		# noise as input => generates images => determines validity
		self.combined = Model(z, valid)
		self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)

	def Generator(self, output_shape):

		# Need to give noise over 1000 time steps
		noise_shape = (1000,100,)

		model = Sequential()
		model.add(LSTM(100, input_shape=noise_shape))
		model.add(Dense(np.prod(output_shape))) # Unclear about this part, but without this cannot produce output dims
		model.add(Reshape(output_shape))

		noise = Input(shape=noise_shape)
		eeg_output = model(noise)

		return Model(noise, eeg_output)

	def Discriminator(self, input_shape):

		model = Sequential()
		model.add(LSTM(100,input_shape=input_shape))
		model.add(Dense(1, activation='sigmoid'))

		inp = Input(shape=input_shape)
		validity = model(inp)

		return Model(inp, validity)


if __name__ == '__main__':

	for i in np.arange(9):
		X, y = parse_eeg_data(i)

		X = np.transpose(X, (0, 2, 1))

		gan = GAN_RNN(X.shape[1:])

		#loss = gan.discriminator.train_on_batch(X, np.ones(X.shape[0]))

		print 'Complete - cannot run to check'
