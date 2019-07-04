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
from joblib import dump

# Directory selection
map_dir = sys.argv[1]
out_dir = sys.argv[2]
name = sys.argv[3]

# today = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
today = datetime.datetime.now().strftime('%Y%m%d')
out_dir += '{}/'.format(today)

try:
    os.makedirs(out_dir)
except:
    pass

# Take the data
l_p = np.load(map_dir + '/test_l_p.npy')
Maps = np.load(map_dir + '/test_Maps.npy')
# l_p2 = np.load(map_dir + '/second_round_l_p.npy')
# Maps2 = np.load(map_dir + '/second_round_Maps.npy')

# l_p = np.concatenate((l_p1, l_p2))
# Maps = np.concatenate((Maps1, Maps2), axis=1)
print('Maps shape :', Maps.shape)

# l_p = np.load(map_dir + "/l_p.npy")
# Maps = np.load(map_dir + "/Maps.npy")

nside = int(np.sqrt(Maps.shape[0]/12))
print("nside = ", nside)

j=0
while 2**j <=  nside :
    j = j+1
if 2**j % nside != 0:
    print("Erreur: Maps has not the good shape: ", Maps.shape, "instead of: ", (len(l_p),12*nside**2))

ecart_Maps = np.sqrt(np.var(Maps))
print("\n Standard deviation of Maps  :", ecart_Maps)

# Data preprocessing Machine Learning
Ntest = np.int(0.1 * len(l_p))
print('Ntest = ', Ntest)

Ntrain = np.int(len(l_p) - Ntest)
print('Ntrain = ', Ntrain)

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
print("\n The shapes of $X_train$, $X_test$ and of the input: ", X_train.shape, shape,
      X_test.shape)
print("\n The shapes of $y_train$ & of $y_test$: ", y_train.shape, y_test.shape)

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

# Callbacks
checkpointer_mse = tf.keras.callbacks.ModelCheckpoint(filepath= out_dir + name + '_weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                      monitor='val_loss',
                                                      verbose=1,
                                                      save_best_only=True,
                                                      save_weights_only=True,
                                                      mode='min',
                                                      period=1)

# stop = kr.callbacks.EarlyStopping(monitor=kr.metrics.mean_absolute_percentage_error,
#                                   min_delta=0,
#                                   patience=10,
#                                   verbose=0,
#                                   mode='auto',
#                                   restore_best_weights=True)

callbacks = [checkpointer_mse]#, stop]

# Model training
model._ckpt_saved_epoch = None
hist = model.fit(X_train, y_train,
                 epochs=6,
                 batch_size=10,
                 validation_split=0.1,
                 verbose=1,
                 callbacks=callbacks,
                 shuffle=True)

error = model.evaluate(X_test, y_test)
print('error :', error)

# Prediction on 100 l_p
prediction = model.predict(X_test)

# Save the model as a pickle in a file
dump(model, out_dir + name + '_model.joblib')

np.save(out_dir + name + '_prediction', prediction)
np.save(out_dir + name + '_hist_loss', hist.history['loss'])
np.save(out_dir + name + '_hist_val_loss', hist.history['val_loss'])