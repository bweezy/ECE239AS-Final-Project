from GAN import GAN
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

print X.shape, y.shape