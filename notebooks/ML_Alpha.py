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
import keras.backend as K
import sys

# Récupération des données
directory = sys.argv[0]
l_p = np.load(directory + "l_p.npy")
Maps = np.load(directory + "Maps.npy")

nside = 16
ecart_Maps = np.sqrt(np.var(Maps))
if Maps.shape != (len(l_p),12*nside**2) :
    print("Erreur: Les maps n'ont pas la bonne shape ", Maps.shape, "au lieu de ", (len(l_p),12*nside**2))
print("\n A titre indicatif l'écart type des valeur des Maps :", ecart_Maps)

#Préparation des données pour le Machine Learning
Ntest = 0.01*len(l_p)
Ntrain = len(l_p)-Ntest
# Attribution des valeurs d'entrées et de sorties pour l'entrainement et les tests
X_train = Maps[:, 0:(Ntrain)]
y_train = l_p[0: (Ntrain)]
X_test = Maps[:, (Ntrain):(Ntrain + Ntest)]
y_test = l_p[(Ntrain) : (Ntrain + Ntest)]

num_classes = 1
# Mise en forme adequate
X_train = X_train.T
X_test = X_test.T
X_train = X_train.reshape(X_train.shape[0], len(X_train[0]), 1)
X_test = X_test.reshape(X_test.shape[0], len(X_test[0]), 1)
shape=(len(Maps[:,0]), 1)
print("\n Les shapes des $X_train$ et des $X_test$ ainsi que celle de l'input attendu: ", X_train.shape, shape, X_test.shape)
print("\n Les shapes des $y_train$ et des $y_test$: ",y_train.shape, y_test.shape)

# NN avec une boucle NBB
inputs=kr.layers.Input(shape)
x=inputs

########### Boucle NBB #################################

for i in range (int(math.log(nside,2))):
#Recog of the neighbours & Convolution
    print(int(nside/(2**(i))), int(nside/(2**(i+1))))
    x = nnhealpix.layers.ConvNeighbours(int(nside/(2**(i))), filters=32, kernel_size=9)(x)
    x = kr.layers.Activation('relu')(x)
#Degrade
    x = nnhealpix.layers.MaxPooling(int(nside/(2**(i))), int(nside/(2**(i+1))))(x)
       
########## Sortie des NBB #############################
     
x = kr.layers.Dropout(0.2)(x)
x = kr.layers.Flatten()(x)
x = kr.layers.Dense(48)(x)
x = kr.layers.Activation('relu')(x)
x = kr.layers.Dense(1)(x)

out=kr.layers.Activation('relu')(x)

# Création et mise en place du model
model = kr.models.Model(inputs=inputs, outputs=out)
model.compile(loss=kr.losses.mse, optimizer='adam', metrics=[kr.metrics.mean_absolute_percentage_error])
model.summary()

# Entrainement du model
hist = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split = 0.1, verbose = 1, shuffle = True)

# Prediction sur 100 l_p
prediction = model.predict(X_test)
diff = prediction-y_test

np.save(directory + "prediction",prediction)
np.save(directory + "diff", diff)
np.save(directory + "hist_loss",hist.history['loss'])
np.save(directory + "hist_val_loss",hist.history['val_loss'])