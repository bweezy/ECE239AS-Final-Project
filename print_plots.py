
import matplotlib.pyplot as plt

import h5py

def print_plots():
    f = h5py.File('train_loss.hdf5', 'r')

    plt.figure(1)
    plt.plot(list(f["d_loss"]))
    plt.title("Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.show()

    plt.figure(2)
    plt.plot(list(f["acc"]))
    plt.title("Coupled Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.show()

    plt.figure(3)
    plt.plot(list(f["g_loss"]))
    plt.title("Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.show()

if __name__ == '__main__':
    print_plots()