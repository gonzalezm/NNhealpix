import math
import keras as kr
import numpy as np
import nnhealpix.layers
import sys
import os
import datetime

# This program is a modified version of ML_Alpha.py
# It takes in args the directories where load and save data
# It is designed to find realistic spectrum from relistic CMB maps
# The spectrum used are temperature power spectrum
# It is also designed to find random spectrum from random maps
# The goal of this is to compare ML to anafast function for smooth data and random data
# It saves data from the training, predictions done by the model and the model itself

# Directory where files will be saved
in_dir = sys.argv[1]
dir = sys.argv[2]
now = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
out_dir = dir + '/{}/'.format(now)

#Create the repository
try:
    os.makedirs(out_dir)
except:
    pass

#Take data
mymaps = np.load(in_dir + "/mymaps.npy")
mycls = np.load(in_dir + "/mycls.npy")
ns = np.sqrt(mymaps.shape[1]/12)
print("Nside = ", ns)

#Take random data
map_rdm = np.load(in_dir + "/map_rdm.npy")
cls_rdm = np.load(in_dir + "/cls_rdm.npy")

#Machine learning
#Data preprocess  

print("The shape of mymaps (inputs): ", mymaps.shape)
print("The shape of mycls / myclsmod (outputs): ", mycls.shape)
num_out = mycls.shape[1] #Number of neurones in output layer
shape=(len(mymaps[0,:]),1) #Shape for the inputs
mymaps = mymaps.reshape(mymaps.shape[0], len(mymaps[0]), 1) #Reshape to be NBB freindly
print("The shape of mymaps (inputs) after reshape: ", mymaps.shape)
nbmodels = mymaps.shape[0] #Found total number of models
print("Number of model: ", nbmodels)
nbtest = int(0.1*nbmodels) #Determine the number of test data among the models
map_rdm = map_rdm.reshape(map_rdm.shape[0], len(map_rdm[0]), 1) #Reshape the random maps to be inputs freindly

#NBB layers
inputs=kr.layers.Input(shape)
x=inputs

#NBB loop (from nside to 1)

for i in range (int(math.log(ns,2))):
    print(int(ns/(2**(i))), int(ns/(2**(i+1))))
#Recog of the neighbours & Convolution
    x = nnhealpix.layers.ConvNeighbours(int(ns/(2**(i))), filters=32, kernel_size=9)(x)
    x = kr.layers.Activation('relu')(x)
#Degrade
    x = nnhealpix.layers.MaxPooling(int(ns/(2**(i))), int(ns/(2**(i+1))))(x)
#End of the NBBs
    
x = kr.layers.Dropout(0.2)(x)
x = kr.layers.Flatten()(x)
x = kr.layers.Dense(48)(x)
x = kr.layers.Activation('relu')(x)
x = kr.layers.Dense(num_out)(x)

out=kr.layers.Activation('relu')(x)

# Creation of the model
mymodel = kr.models.Model(inputs=inputs, outputs=out)
mymodel.compile(loss=kr.losses.mse, optimizer='adam',
                metrics=[kr.metrics.mean_absolute_percentage_error])

mymodel.summary()

# Callbacks
checkpointer_mse = kr.callbacks.ModelCheckpoint(filepath = out_dir + now + '_weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                monitor = 'val_loss',
                                                verbose = 1,
                                                save_best_only = True,
                                                save_weights_only = True,
                                                mode = 'min',
                                                period = 1)

# stop = kr.callbacks.EarlyStopping(monitor=kr.metrics.mean_absolute_percentage_error,
#                                   patience=10,
#                                   verbose=0,
#                                   restore_best_weights=True)

callbacks = [checkpointer_mse]  # , stop]

# Training
hist = mymodel.fit(mymaps[:(nbmodels-nbtest),:,:],
                          mycls[:(nbmodels-nbtest),:],
                          epochs=100, batch_size=32,
                          validation_split = 0.1,
                          verbose = 1,
                          callbacks=callbacks,
                          shuffle = True)

#Creation of the random model
model_rdm = kr.models.Model(inputs=inputs, outputs=out)
model_rdm.compile(loss=kr.losses.mse,
                  optimizer='adam',
                  metrics=[kr.metrics.mean_absolute_percentage_error])

model_rdm.summary()

# Callbacks for random model
checkpointer_mse_rdm = kr.callbacks.ModelCheckpoint(filepath = out_dir + now + '_weights_rdm.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                monitor = 'val_loss',
                                                verbose = 1,
                                                save_best_only = True,
                                                save_weights_only = True,
                                                mode = 'min',
                                                period = 1)

# stop_rdm = kr.callbacks.EarlyStopping(monitor=kr.metrics.mean_absolute_percentage_error,
#                                   patience=10,
#                                   verbose=0,
#                                   restore_best_weights=True)

callbacks_rdm = [checkpointer_mse_rdm]  # , stop_rdm]


#random model training
hist_rdm=model_rdm.fit(map_rdm[:(nbmodels-nbtest),:,:],
                               cls_rdm[:(nbmodels-nbtest),:],
                               epochs=100, batch_size=32,
                               validation_split = 0.1,
                               verbose = 1,
                               callbacks=callbacks_rdm,
                               shuffle = True)

#Make a prédiction on smooth model with random and smooth data
prediction = mymodel.predict(mymaps[(nbmodels-nbtest):,:,:])
prediction2 = mymodel.predict(map_rdm[(nbmodels-nbtest):,:,:])

#Prediction on random model with random and smooth data
pred_rdm = model_rdm.predict(mymaps[(nbmodels-nbtest):,:,:])
pred_rdm2 = model_rdm.predict(map_rdm[(nbmodels-nbtest):,:,:])

#Saves models and data
kr.models.save_model(mymodel, out_dir + now + "_mymodel.h5py.File")
kr.models.save_model(model_rdm, out_dir + now + "_model_rdm.h5py.File")
np.save(out_dir + now + "_prediction", prediction)
np.save(out_dir + now + "_prediction2", prediction2)
np.save(out_dir + now + "_pred_rdm", pred_rdm)
np.save(out_dir + now + "_pred_rdm2", pred_rdm2)
np.save(out_dir + now + "_loss-alm", hist.history['loss'])
np.save(out_dir + now + "_val_loss-alm", hist.history['val_loss'])
np.save(out_dir + now + "_loss-alm_rdm", hist_rdm.history['loss'])
np.save(out_dir + now + "_val_loss-alm_rdm", hist_rdm.history['val_loss'])