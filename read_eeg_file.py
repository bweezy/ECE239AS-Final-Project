import numpy as np
import h5py

data_directory = '../239as-eeg-data'

files = ['A01T_slice.mat', 'A02T_slice.mat', 'A03T_slice.mat',
         'A04T_slice.mat', 'A05T_slice.mat', 'A06T_slice.mat',
         'A07T_slice.mat', 'A08T_slice.mat', 'A09T_slice.mat']

def parse_eeg_data(subject_number):
  
  data = h5py.File(data_directory + '/' + files[subject_number], 'r')
  X = np.copy(data['image'])
  y = np.copy(data['type'])
  y = y[0,0:X.shape[0]:1]
  y = np.asarray(y, dtype=np.int32)

  X = X[:,:22,:]
  y = y - 768

  nan_trials = []

  for i in np.arange(X.shape[0]):
    if np.isnan(np.sum(X[i])):
      nan_trials.append(i)

  X = np.delete(X,nan_trials,0)
  y = np.delete(y,nan_trials,0)

  X -= np.mean(X)
  X /= np.max(np.abs(X))

  return X, y


def parse_eeg_data_for_GAN(subject_number):

	X, y = parse_eeg_data(subject_number)

	complete = np.copy(X)
	incomplete = np.delete(X, 9, 1)

	return incomplete, complete
