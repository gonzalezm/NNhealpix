# -*- coding: utf-8 -*-
# %matplotlib inline
import math
import keras as kr
import numpy as np
import nnhealpix.layers
import sys
import os
import datetime

# This program is a ML program
# It takes in args the directories where load and save data
# It takes also the name to give to data
# It is design to find the mean of a gaussian spectrum from a CMB map based on gaussian spectrum
# It saves data from training, prediction done by the model and the model itself

#Take in arguments path where find the data and path where save new data. Take the name given during the creation of data
name_in = sys.argv[1]
in_dir = sys.argv[2]
out_dir = sys.argv[3]

#Add date and time to the path to save to avoid "same name file" problems.
today = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
out_dir += '/{}/'.format(today)

#Creat the repository where save new data
try:
    os.makedirs(out_dir)
except:
    pass

# Load the data
l_p = np.load(in_dir + "/" + name_in + "_l_p.npy")
Maps = np.load(in_dir + "/" + name_in + "_Maps.npy")

# In case you need to concatenate many files :
# l_p = np.concatenate((l_p1, l_p2))
# Maps = np.concatenate((Maps1, Maps2), axis=1)
print('Maps shape :', Maps.shape)

#Find Nside from maps shape
nside = int(np.sqrt(Maps.shape[0] / 12))
print("nside = ", nside)

#Check the maps have the shape expected
j = 0
while 2 ** j <= nside:
    j = j + 1
if 2 ** j % nside != 0:
    print("Erreur: Maps has not the good shape: ", Maps.shape, "instead of: ", (len(l_p), 12 * nside ** 2))

ecart_Maps = np.sqrt(np.var(Maps))
print("\n Standard deviation of Maps  :", ecart_Maps)

# Data preprocessing Machine Learning
Ntest = np.int(0.1 * len(l_p))
print('Ntest = ', Ntest)

Ntrain = len(l_p) - Ntest
print('Ntrain = ', Ntrain)

# Split between train and test
X_train = Maps[:, 0:Ntrain]
y_train = l_p[0: Ntrain]
X_test = Maps[:, Ntrain:(Ntrain + Ntest)]
y_test = l_p[Ntrain: (Ntrain + Ntest)]

num_classes = 1
# Changing the shape for NBB (NBB is the special part of the Neural Network used)
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

# NBB loop (conv and degrade from nside to 1)

for i in range(int(math.log(nside, 2))):
    # Recog of the neighbours & Convolution
    print(int(nside / (2 ** i)), int(nside / (2 ** (i + 1))))
    x = nnhealpix.layers.ConvNeighbours(int(nside / (2 ** i)),
                                        filters=32,
                                        kernel_size=9)(x)
    x = kr.layers.Activation('relu')(x)
    # Degrade
    x = nnhealpix.layers.MaxPooling(int(nside / (2 ** i)),
                                    int(nside / (2 ** (i + 1))))(x)

# End of the NBBs

x = kr.layers.Dropout(0.2)(x)
x = kr.layers.Flatten()(x)
x = kr.layers.Dense(48)(x)
x = kr.layers.Activation('relu')(x)
x = kr.layers.Dense(1)(x)

out = kr.layers.Activation('relu')(x)

# Creation of the model
model = kr.models.Model(inputs=inputs, outputs=out)
model.compile(loss=kr.losses.mse,
              optimizer='adam',
              metrics=[kr.metrics.mean_absolute_percentage_error])
model.summary()

# Callbacks
checkpointer_mse = kr.callbacks.ModelCheckpoint(filepath=out_dir + today + '_weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                monitor='val_loss',
                                                verbose=1,
                                                save_best_only=True,
                                                save_weights_only=True,
                                                mode='min',
                                                period=1)

stop = kr.callbacks.EarlyStopping(monitor=kr.metrics.mean_absolute_percentage_error, patience=10)

callbacks = [checkpointer_mse, stop]

# Model training
# model._ckpt_saved_epoch = None
hist = model.fit(X_train, y_train,
                 epochs=50,
                 batch_size=32,
                 validation_split=0.1,
                 verbose=1,
                 callbacks=callbacks,
                 shuffle=True)

error = model.evaluate(X_test, y_test)
print('error :', error)

# Prediction on 100 l_p
prediction = model.predict(X_test)

# Save the model as a pickle in a file
kr.models.save_model(model, out_dir + today + '_model.h5py.File')

np.save(out_dir + today + '_prediction', prediction)
np.save(out_dir + today + '_hist_loss', hist.history['loss'])
np.save(out_dir + today + '_hist_val_loss', hist.history['val_loss'])
