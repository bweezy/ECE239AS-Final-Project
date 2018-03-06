from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Input, LSTM, Dense, Flatten



import numpy as np
import h5py


class GAN():

	def __init__(self, use_reccurent_gen=False,
								use_recurrent_disc=False):
		self.input_size = 1000
		self.channels = 22

		self.optimizer = Adam(lr=1e-3, beta_1=.9, beta_2=.99,
													decay=.99)

		self.generator = self.Generator(recurrent=use_reccurent_gen)


		self.discriminator = self.Discriminator(recurrent=use_recurrent_disc)
		self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
		


	def Generator(self, recurrent):
		
		# Look into size of noise vectors effect on generative models
		data_shape = (self.channels, self.input_size)
		noise_shape = (100,)

		model = Sequential()
		model.add(LSTM(100, input_shape=noise_shape))
		model.add(Reshape(data_shape))

		noise = Input(shape=noise_shape)
		eeg_output = model(noise)

		return Model(inp, eeg_output)

	def Discriminator(self, recurrent):

		data_shape = (self.channels, self.input_size)

		model = Sequential()
		model.add(LSTM(100,input_shape=data_shape))
		model.add(Dense(1, activation='sigmoid'))

		inp = Input(shape=data_shape)
		validity = model(inp)

		return Model(inp, validity)

	def train(self, X, y, epochs, batch_size=128, save_interval=50):

		X_train = X
		y_train = y

		half_batch = int(batch_size / 2)

		print self.discriminator.train_on_batch(X_train, np.ones(X.shape[0]))

		# for epoch in range(epochs):
		# 	idx = np.random.randint(0, X_train.shape[0], half_batch)
		# 	readings = X_train[idx]
        #
		# 	noise = np.random.normal(0, 1, (half_batch, 100)) # this is not correct -- needs to be similar to eeg values
        #
		# 	gen_readings = noise # this is not correct -- generator should make a prediction
        #
		# 	d_loss_real = self.discriminator.fit(readings, np.ones((half_batch, 288)))
