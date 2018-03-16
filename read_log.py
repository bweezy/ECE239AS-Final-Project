import numpy as np
import h5py

if __name__ == '__main__':
	f = h5py.File('train_loss.hdf5', 'r')
	print(f['d_loss'][:])
	print(f['acc'][:])
	print(f['g_loss'][:])
