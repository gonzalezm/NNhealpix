# -*- coding: utf-8 -*-
# %matplotlib inline
import tensorflow as tf
import math
import keras as kr
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from keras.models import Sequential
import nnhealpix as nn
import nnhealpix.layers
import sys
import os
import datetime

# Directory selection
map_dir = sys.argv[1]
out_dir = sys.argv[2]
today = datetime.datetime.now().strftime('%Y%m%d')
out_dir += '{}/'.format(today)

try:
    os.makedirs(out_dir)
except:
    pass

# Take the data
l_p = np.load(map_dir + "/l_p.npy")
Maps = np.load(map_dir + "/Maps.npy")

nside = 16
ecart_Maps = np.sqrt(np.var(Maps))
if Maps.shape != (len(l_p), 12 * nside ** 2):
    print("Erreur: Les maps n'ont pas la bonne shape ", Maps.shape, "au lieu de ", (len(l_p), 12 * nside ** 2))
print("\n A titre indicatif l'écart type des valeur des Maps :", ecart_Maps)

# Data preprocessing Machine Learning
Ntest = 0.01 * len(l_p)
Ntrain = len(l_p) - Ntest
# Split between train and test
X_train = Maps[:, 0:(Ntrain)]
y_train = l_p[0: (Ntrain)]
X_test = Maps[:, (Ntrain):(Ntrain + Ntest)]
y_test = l_p[(Ntrain): (Ntrain + Ntest)]

num_classes = 1
# Changing the shape for NBB
X_train = X_train.T
X_test = X_test.T
X_train = X_train.reshape(X_train.shape[0], len(X_train[0]), 1)
X_test = X_test.reshape(X_test.shape[0], len(X_test[0]), 1)
shape = (len(Maps[:, 0]), 1)
print("\n Les shapes des $X_train$ et des $X_test$ ainsi que celle de l'input attendu: ", X_train.shape, shape,
      X_test.shape)
print("\n Les shapes des $y_train$ et des $y_test$: ", y_train.shape, y_test.shape)

# NN with NBB loop
inputs = kr.layers.Input(shape)
x = inputs

# NBB loop

for i in range(int(math.log(nside, 2))):
    # Recog of the neighbours & Convolution
    print(int(nside / (2 ** (i))), int(nside / (2 ** (i + 1))))
    x = nnhealpix.layers.ConvNeighbours(int(nside / (2 ** (i))), filters=32, kernel_size=9)(x)
    x = kr.layers.Activation('relu')(x)
    # Degrade
    x = nnhealpix.layers.MaxPooling(int(nside / (2 ** (i))), int(nside / (2 ** (i + 1))))(x)

# End of the NBBs

x = kr.layers.Dropout(0.2)(x)
x = kr.layers.Flatten()(x)
x = kr.layers.Dense(48)(x)
x = kr.layers.Activation('relu')(x)
x = kr.layers.Dense(1)(x)

out = kr.layers.Activation('relu')(x)

# Creation of the model
model = kr.models.Model(inputs=inputs, outputs=out)
model.compile(loss=kr.losses.mse, optimizer='adam', metrics=[kr.metrics.mean_absolute_percentage_error])
model.summary()

# Model training
hist = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1, shuffle=True)

# Prediction on 100 l_p
prediction = model.predict(X_test)
diff = prediction - y_test

np.save(out_dir + "/prediction", prediction)
np.save(out_dir + "/diff", diff)
np.save(out_dir + "/hist_loss", hist.history['loss'])
np.save(out_dir + "/hist_val_loss", hist.history['val_loss'])
