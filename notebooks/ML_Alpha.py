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

directory = input("Choose a directory where take the data (None = same folder than this program): ")
l_p = np.load(directory + "l_p.npy")

Maps = np.load(directory + "Maps.npy")
moy_Maps = np.mean(Maps)
ecart_Maps = np.sqrt(np.var(Maps))
Max_Maps = np.abs(Maps).max()
print(Maps.shape)
print(ecart_Maps)
print(Max_Maps)
print(Maps)

nside = 16
Ntest = 100
Ntrain = len(l_p)-Ntest
# Normalisation des sorties
Max_l_p=np.abs(l_p).max()
NNl_p = (l_p-5)/(50.0 - 5.0)
# Attribution des valeurs d'entrées et de sorties pour l'entrainement et les tests
X_train = Maps[:, 0:(Ntrain)]
y_train = l_p[0: (Ntrain)]
X_test = Maps[:, (Ntrain):(Ntrain + Ntest)]
y_test = l_p[(Ntrain) : (Ntrain + Ntest)]

print(y_train.shape, y_test.shape)
print(np.abs(NNl_p).max())
print(y_train)

seed = 7
np.random.seed(seed)
shape=(len(Maps[:,0]), 1)
num_classes = 1 #y_train.shape[1]
# Mise en forme adequate
X_train = X_train.T
X_test = X_test.T
print(X_train.shape, shape, X_test.shape)
X_train = X_train.reshape(X_train.shape[0], len(X_train[0]), 1)
X_test = X_test.reshape(X_test.shape[0], len(X_test[0]), 1)
print(X_train.shape, shape, X_test.shape)

inputs=kr.layers.Input(shape)
x=inputs
for i in range (int(math.log(nside,2))):
#Recog of the neighbours & Convolution
    print(int(nside/(2**(i))), int(nside/(2**(i+1))))
    x = nnhealpix.layers.ConvNeighbours(int(nside/(2**(i))), filters=32, kernel_size=9)(x)
    x = kr.layers.Activation('relu')(x)
#Degrade
    x = nnhealpix.layers.MaxPooling(int(nside/(2**(i))), int(nside/(2**(i+1))))(x)
#Sortie des NBB
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
#scores = model.evaluate(X_test, y_test, verbose=0)
#print("CNN Error: %.2f%%" % (100-scores[1]*100))

prediction = model.predict(X_test)
print(prediction.shape)

print(prediction)

prediction.shape
diff = prediction-y_test

# Comparaison entre les valeurs obtenues et les valeurs recherchées
fig1 = plt.figure(1, figsize = (10, 10), edgecolor = 'black')
plt.ylabel('$\ell _p$ recherchés et $\ell _p$ prédit')
plt.xlabel('')
plt.title('Comparatif des données recherchées et prédites par le modèle')
for i in range (len(prediction[:,0])):
    plt.plot(i, prediction[i,0], color ='blue',marker='o')
    plt.plot(i, l_p[i], color ='red',marker='+')
plt.legend(['$\ell _p$ prédits','$\ell _p$ recherchés' ],loc = 2)
plt.show()

fig2 = plt.figure(2, figsize = (10, 10), edgecolor = 'black')
plt.ylabel('Différence entre les $\ell _p$ prédits et les $\ell _p$ recherché')
plt.xlabel('')
plt.title('Différence entre la prédiction et les données attendues.')
for i in range (len(diff)):
    plt.plot(i, diff[i,0], color ='green',marker='x')
plt.show()

plt.plot(l_p[Ntrain:(Ntrain+Ntest)], prediction[:, 0], '.')
