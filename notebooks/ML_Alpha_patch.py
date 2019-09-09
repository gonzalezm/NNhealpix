import math
import keras as kr
import numpy as np
import healpy as hp
import nnhealpix.layers
import sys
import os
import datetime

# This program is a modified version of ML_Alpha-alm.py
# The goal is to make a ML program which find Power Spectrum from a patch of the CMB
# It is for Temperature power spectrum
# It saves predictions on the model, the model itself and data from the training
# It takes in args directory where load and where save data

# Directory where files will be saved
map_dir = sys.argv[1]
dir = sys.argv[2]
now = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
out_dir = dir + '/{}/'.format(now)

try:
    os.makedirs(out_dir)
except:
    pass

# Take data
mymaps = np.load(map_dir + "/mymaps.npy")
myclsmod = np.load(map_dir + "/myclsmod.npy")
ns = np.sqrt(mymaps.shape[1] / 12)
print("Nside = ", ns)

# define the zone of the patch
radius = 20  # deg
theta = 3 * np.pi / 4
phy = np.pi / 2
vec = hp.ang2vec(theta, phy)

# make the map with only the patch from full maps
patch = hp.query_disc(ns, vec, radius=np.radians(radius))
map_patch = np.full((mymaps.shape[0], mymaps.shape[1]), hp.UNSEEN)
for i in range(mymaps.shape[0]):
    map_patch[i, patch] = mymaps[i, patch]

# Machine learning
# Data preprocess

print("The shape of map_patch (inputs): ", map_patch.shape)
print("The shape of mycls / myclsmod (outputs): ", myclsmod.shape)
num_out = myclsmod.shape[1]  # find the number of neurones in the ouput layer
shape = (len(mymaps[0, :]), 1)  # give the shape NBB inputs freindly
map_patch = map_patch.reshape(map_patch.shape[0], len(map_patch[0]), 1)  # Reshape to be inputs freindly
print("The shape of map_patch (inputs) after reshape: ", map_patch.shape)
nbmodels = map_patch.shape[0]  # find the number of models
print("Number of model: ", nbmodels)
nbtest = int(0.1 * nbmodels)  # determine the number of test among models

# NBB layers
inputs = kr.layers.Input(shape)
x = inputs

# NBB loop (from nside to 1)

for i in range(int(math.log(ns, 2))):
    print(int(ns / (2 ** i)), int(ns / (2 ** (i + 1))))
    # Recog of the neighbours & Convolution
    x = nnhealpix.layers.ConvNeighbours(int(ns / (2 ** i)), filters=32, kernel_size=9)(x)
    x = kr.layers.Activation('relu')(x)
    # Degrade
    x = nnhealpix.layers.MaxPooling(int(ns / (2 ** i)), int(ns / (2 ** (i + 1))))(x)
# End of the NBBs

x = kr.layers.Dropout(0.2)(x)
x = kr.layers.Flatten()(x)
x = kr.layers.Dense(48)(x)
x = kr.layers.Activation('relu')(x)
x = kr.layers.Dense(num_out)(x)

out = kr.layers.Activation('relu')(x)

# Creation of the model
model_patch = kr.models.Model(inputs=inputs, outputs=out)
model_patch.compile(loss=kr.losses.mse,
                    optimizer='adam',
                    metrics=[kr.metrics.mean_absolute_percentage_error])

model_patch.summary()

# Callbacks
checkpointer_mse = kr.callbacks.ModelCheckpoint(filepath=out_dir + now + '_weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                monitor='val_loss',
                                                verbose=1,
                                                save_best_only=True,
                                                save_weights_only=True,
                                                mode='min',
                                                period=1)

# stop = kr.callbacks.EarlyStopping(monitor=kr.metrics.mean_absolute_percentage_error,
#                                   patience=10,
#                                   verbose=0,
#                                   restore_best_weights=True)

callbacks = [checkpointer_mse]  # , stop]

# Training
hist = model_patch.fit(map_patch[:(nbmodels - nbtest), :, :],
                       myclsmod[:(nbmodels - nbtest), :],
                       epochs=100, batch_size=32,
                       validation_split=0.1,
                       verbose=1,
                       callbacks=callbacks,
                       shuffle=True)

# Make a prédiction
prediction = model_patch.predict(map_patch[(nbmodels - nbtest):, :, :])

kr.models.save_model(model_patch, out_dir + now + "_mymodel.joblib")
np.save(out_dir + now + "_prediction", prediction)
np.save(out_dir + now + "_loss-alm", hist.history['loss'])
np.save(out_dir + now + "_val_loss-alm", hist.history['val_loss'])
