from keras.models import Sequential, Model
from keras.optimizers import Adam



import numpy as np
import h5py


class GAN():

	def __init__(self, use_reccurent_gen=False,
								use_recurrent_disc=False):
		self.input_size = None
		self.channels = None

		self.optimizer = Adam(lr=1e-3, beta_1=.9, beta_2=.99,
													decay=.99)


		self.generator = self.Generator(recurrent=use_reccurent_gen)
		self.discriminator = self.Discriminator(recurrent=use_recurrent_disc)
		


	def Generator(self, recurrent):
		
		# Look into size of noise vectors effect on generative models
		input_shape = (100,)

		model = Sequential()




	def Discriminator(self, recurrent):


		

		# Assuming model is not recurrent

		model = Sequential()

