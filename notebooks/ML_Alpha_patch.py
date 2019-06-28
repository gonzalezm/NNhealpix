import tensorflow as tf
import math
import keras as kr
import numpy as np
from keras.models import Sequential
import healpy as hp
import nnhealpix as nn
import nnhealpix.layers
from pylab import *
import sys
import os
import datetime
from joblib import dump

# Directory where files will be saved
map_dir = sys.argv[1]
dir = sys.argv[2]
now = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
out_dir = dir + '{}/'.format(now)

try:
    os.makedirs(out_dir)
except:
    pass

#Take data
mymaps = np.load(map_dir + "/mymaps.npy")
myclsmod = np.load(map_dir + "/myclsmod.npy")
ns = np.sqrt(mymaps.shape[1]/12)
print("Nside = ", ns)

radius = 20 #deg
theta = 3*np.pi/4
phy = np.pi/2
vec=hp.ang2vec(theta, phy)

patch = hp.query_disc(ns, vec, radius=np.radians(radius))
map_patch = np.full((mymaps.shape[0],mymaps.shape[1]), hp.UNSEEN)
for i in range (mymaps.shape[0]):
    map_patch[i, patch] = mymaps[i, patch]


#Machine learning
#Data preprocess  

print("La shape de map_patch (inputs): ", map_patch.shape)
print("La shape de mycls / myclsmod (outputs): ", myclsmod.shape)
shape=(len(map_patch[0,:]),1)
map_patch = map_patch.reshape(map_patch.shape[0], len(map_patch[0]), 1)
nbmodels = map_patch.shape[1]
nbtest = int(0.1*nbmodels)

#NBB layers
inputs=kr.layers.Input(shape)
x=inputs
for i in range (int(math.log(ns,2))):
#Recog of the neighbours & Convolution
    print(int(ns/(2**(i))), int(ns/(2**(i+1))))
    x = nnhealpix.layers.ConvNeighbours(int(ns/(2**(i))), filters=32, kernel_size=9)(x)
    x = kr.layers.Activation('relu')(x)
#Degrade
    x = nnhealpix.layers.MaxPooling(int(ns/(2**(i))), int(ns/(2**(i+1))))(x)
#End of the NBBs
x = kr.layers.Dropout(0.2)(x)
x = kr.layers.Flatten()(x)
x = kr.layers.Dense(48)(x)
x = kr.layers.Activation('relu')(x)
x = kr.layers.Dense(32)(x)

out=kr.layers.Activation('relu')(x)

# Creation of the model
mymodel = kr.models.Model(inputs=inputs, outputs=out)
mymodel.compile(loss=kr.losses.mse, optimizer='adam', metrics=[kr.metrics.mean_absolute_percentage_error])
mymodel.summary()

# Training
hist = mymodel.fit(map_patch[:(nbmodels-nbtest),:,:], myclsmod[:(nbmodels-nbtest),:], epochs=5, batch_size=32, validation_split = 0.1, verbose = 1, shuffle = True)

#Make a prédiction
prediction=mymodel.predict(map_patch[(nbmodels-nbtest):,:,:])

dump(mymodel, out_dir + "/mymodel.joblib")
np.save(out_dir + "/loss-alm", hist.history['loss'])
np.save(out_dir + "/val_loss-alm", hist.history['val_loss'])