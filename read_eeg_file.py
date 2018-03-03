import numpy as np
import h5py



def parse_eeg_data(filepath):

	print filepath
	data = h5py.File(filepath, 'r')
	X = np.copy(data['image'])
	y = np.copy(data['type'])
	y = y[0,0:X.shape[0]:1]
	y = np.asarray(y, dtype=np.int32)

	return X, y
