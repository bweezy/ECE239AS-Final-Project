from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Input, LSTM, Dense, Flatten, Reshape, Concatenate, Lambda
from keras import backend

from read_eeg_file import parse_eeg_data_for_GAN

import numpy as np
import h5py


def EEG_Concatenate(input_list):
	input1, x = input_list

	slice1 = input1[:,:,:9]
	slice2 = input1[:,:,9:]

	slice1 = backend.permute_dimensions(slice1, (0, 2, 1))
	slice2 = backend.permute_dimensions(slice2, (0, 2, 1))
	x = backend.expand_dims(x, axis=1)

	output = Concatenate(axis=1)([slice1, x, slice2])
	return output

class GAN_RNN():

	def __init__(self, gen_input_shape, disc_input_shape):
		self.optimizer = Adam(lr=1e-3, beta_1=.9, beta_2=.99, decay=.99)

		self.generator = self.Generator(input_shape=gen_input_shape)
		# self.generator.compile(loss='binary_crossentropy', optimizer=self.optimizer)

		self.discriminator = self.Discriminator(input_shape=disc_input_shape)
		self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

		# The generator takes noise as input and generated imgs
		z = Input(shape=gen_input_shape)
		gen_output = self.generator(z)

		# For the combined model we will only train the generator
		self.discriminator.trainable = False

		# The valid takes generated images as input and determines validity
		valid = self.discriminator(gen_output)

		# The combined model  (stacked generator and discriminator) takes
		# noise as input => generates images => determines validity
		self.combined = Model(z, valid)
		self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)

	def Generator(self, input_shape):

		# Need to give noise over 729 time steps - input shape is 729 x 21
		input1 = Input(shape=input_shape)
		t, c = input_shape

		intermediate_model = LSTM(100, input_shape=input_shape)(input1)
		intermediate_model_out = Dense(t)(intermediate_model)

		gen_output = Lambda(EEG_Concatenate, output_shape=(c + 1, t))([input1, intermediate_model_out])

		model = Model(inputs=input1, outputs=gen_output)

		return model

	def Discriminator(self, input_shape):

		model = Sequential()
		model.add(LSTM(100,input_shape=input_shape))
		model.add(Dense(1, activation='sigmoid'))

		inp = Input(shape=input_shape)
		validity = model(inp)

		return Model(inp, validity)


if __name__ == '__main__':

	for i in np.arange(9):
		incomplete, complete = parse_eeg_data_for_GAN(0)

		incomplete = incomplete[:, :, :729]
		complete = complete[:, :, :729]

		incomplete = np.transpose(incomplete, (0, 2, 1))

		gan = GAN_RNN(gen_input_shape=incomplete.shape[1:], disc_input_shape=complete.shape[1:])

		#loss = gan.discriminator.train_on_batch(X, np.ones(X.shape[0]))

		print 'Complete - cannot run to check'
