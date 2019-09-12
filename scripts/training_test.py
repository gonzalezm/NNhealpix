import healpy as hp
import numpy as np
import os
import sys
import datetime
import sklearn.model_selection as skmodel

import ConvNNTempLib as cnn

"""
Test script to find the maximum of a gaussian power spectrum
parameters: input directory, output directory
"""
in_dir = sys.argv[1]
out_dir = sys.argv[2]

today = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
out_dir += '/output-{}/'.format(today)

# Creat the repository where save new data
os.makedirs(out_dir, exist_ok=True)

# Load the data
lp = np.load(in_dir + '/lp.npy')
maps = np.load(in_dir + '/maps.npy')
print('maps shape :', maps.shape)

# Get Nside
nside = hp.npix2nside(maps.shape[1])
print('nside : ', nside)

# Add noise and normalize the maps
sigma_n = 0.
maps = cnn.AddWhiteNoise(maps, sigma_n)
maps = cnn.NormalizeMaps(maps)

# Split train and test datas
X_train, X_test, y_train, y_test = skmodel.train_test_split(maps, lp, test_size=0.1)

# Make a model
model = cnn.make_model(nside, y_train[0].size)

# Train the model
model, hist = cnn.make_training(model, X_train, y_train, 0.1, 10, 20, out_dir, today=today)

X_test = np.expand_dims(X_test, axis=2)

error = model.evaluate(X_test, y_test)

# Print the final error over validation data    
print('error:', error)
