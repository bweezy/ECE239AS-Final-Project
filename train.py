from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np


def train(self, epochs, batch_size=128, save_interval=50):

	# Load data set

	# Rescale data set (zero mean, unit variance)

	half_batch = int(batch_size / 2)

	for epoch in range(epochs):

		# -------------------
		# Train Discriminator
		# -------------------

		# random half batch implementaiton?

        # Select a random half batch of images
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        imgs = X_train[idx]

		# Caluclate Losses


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
        if (epoch % save_interval == 0) self.save_imgs(epoch)


    plt.figure(2)
    plt.subplot(211)
    plt.plot(gen_eeg[0, 9])
    plt.subplot(212)
    plt.plot(real_eeg[0, 9])
    plt.show()


