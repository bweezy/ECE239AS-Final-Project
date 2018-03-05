from GAN_CNN import GAN_CNN
from read_eeg_file import parse_eeg_data
import numpy as np 
import h5py


data_directory = './project_datasets'

files = ['A01T_slice.mat', 'A02T_slice.mat', 'A03T_slice.mat',
				 'A04T_slice.mat', 'A05T_slice.mat', 'A06T_slice.mat',
				 'A07T_slice.mat', 'A08T_slice.mat', 'A09T_slice.mat']

X, y = parse_eeg_data(data_directory + '/' + files[0])

X = X[:,:22,:]
y = y - 768

nan_trials = []

for i in np.arange(X.shape[0]):
	if np.isnan(np.sum(X[i])):
		nan_trials.append(i)

X = np.delete(X,nan_trials,0)
y = np.delete(y,nan_trials,0)

print X.shape, y.shape

X = X[:,:,:, np.newaxis]

#print X

#X_train = X[0:60]
gan = GAN_CNN(X.shape[1:])

loss = gan.discriminator.train_on_batch(X, np.ones(X.shape[0]))

print loss
