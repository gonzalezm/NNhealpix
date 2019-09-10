import keras as kr
import healpy as hp
import numpy as np
import sys
import os
import datetime

import ConvNNTempLib as cnn


# This program is designed to find complete TT spectrum from CMB maps
# The goal of this is to compare ML to anafast function for smooth data and random data
# It saves data from the training, predictions done by the model and the model itself

# Arguments
name_in = sys.argv[1] # Name given to data you want to load
in_dir = sys.argv[2] # Directory path for loading data
out_dir = sys.argv[3] # Directory path for saving results

# Add date and time to the path to save to avoid "same name file" problems.
today = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
out_dir += '/{}/'.format(today)

# Creat the repository where save new data
try:
    os.makedirs(out_dir)
except:
    pass

# Load the data
C_l = np.load(in_dir + "/" + name_in + "_C_l.npy")
Maps = np.load(in_dir + "/" + name_in + "_Maps.npy")

# Get Nside
Nside = hp.npix2nside(Maps.shape[1])

# Add noise and normalize the maps
sigma_n = 0.
Maps = cnn.AddWhiteNoise(Maps, sigma_n)
Maps = cnn.NormalizeMaps(Maps)

# Split train/test data
Ntest = 0.1
Ntrain = 1 - Ntest
X_train, y_train, X_test, y_test, num_out, shape = cnn.PreprocessML(Maps, C_l, Ntest, Ntrain)

# Build all layers as it is in the paper
inputs, out = cnn.ConvNNhealpix(shape, Nside, num_out)

# Creation of the model and training
model, hist, loss, val_loss = cnn.MakeAndTrainModel(X_train, y_train,
                                                    X_test, y_test,
                                                    epoch=20, batch_size=32,
                                                    out_dir=out_dir, today=today,
                                                    inputs=inputs, out=out,
                                                    retrain=False)

# Print the final error
error = model.evaluate(X_test, y_test)
print('error :', error)

# Save the model as a pickle in a file
kr.models.save_model(model, out_dir + today + '_model.h5py.File')

# Save results
np.save(out_dir + today + '_hist_loss', hist.history['loss'])
np.save(out_dir + today + '_hist_val_loss', hist.history['val_loss'])
np.save(out_dir + today + '_history', hist.history)
