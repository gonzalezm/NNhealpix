import healpy as hp
import numpy as np
import os
import sys
import datetime
import sklearn.model_selection as skmodel

import ConvNNTempLib as cnn

name = sys.argv[1]
dir = sys.argv[2]
input_name = sys.argv[3]

in_dir = dir + '/' + input_name

# Directory where files will be saved
today = datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')
out_dir = dir + '/output-{}/'.format(today)

# Creat the repository where save new data
os.makedirs(out_dir, exist_ok=True)

# Load the data
lp = np.load(in_dir + '/_lp.npy')
maps = np.load(in_dir + '/_maps.npy')
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
model = cnn.make_model(nside, 1)

# Train the model
model, hist = cnn.make_training(model, X_train, y_train, 0.1, 10, 20, out_dir, today=today)

# Print the final error
print('loss :', hist.history['loss'])
