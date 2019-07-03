import math
import keras as kr
import numpy as np
import nnhealpix.layers
import sys
import os
import datetime
from joblib import dump

# Directory where files will be saved
in_dir = sys.argv[1]
dir = sys.argv[2]
now = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
out_dir = dir + '{}/'.format(now)

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
num_out = mycls.shape[1]
shape=(len(mymaps[0,:]),1)
mymaps = mymaps.reshape(mymaps.shape[0], len(mymaps[0]), 1)
print("The shape of mymaps (inputs) after reshape: ", mymaps.shape)
nbmodels = mymaps.shape[0]
print("Number of model: ", nbmodels)
nbtest = int(0.1*nbmodels)
map_rdm = map_rdm.reshape(map_rdm.shape[0], len(map_rdm[0]), 1)

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
x = kr.layers.Dense(num_out)(x)

out=kr.layers.Activation('relu')(x)

# Creation of the model
mymodel = kr.models.Model(inputs=inputs, outputs=out)
mymodel.compile(loss=kr.losses.mse, optimizer='adam', metrics=[kr.metrics.mean_absolute_percentage_error])
mymodel.summary()

# Training
hist = mymodel.fit(mymaps[:(nbmodels-nbtest),:,:], mycls[:(nbmodels-nbtest),:], epochs=15, batch_size=32, validation_split = 0.1, verbose = 1, shuffle = True)

#Random model
model_rdm = kr.models.Model(inputs=inputs, outputs=out)
model_rdm.compile(loss=kr.losses.mse, optimizer='adam', metrics=[kr.metrics.mean_absolute_percentage_error])
model_rdm.summary()

#random training
hist_rdm=model_rdm.fit(map_rdm[:(nbmodels-nbtest),:,:], cls_rdm[:(nbmodels-nbtest),:], epochs=15, batch_size=32, validation_split = 0.1, verbose = 1, shuffle = True)

#Make a prédiction
prediction = mymodel.predict(mymaps[(nbmodels-nbtest):,:,:])
prediction2 = mymodel.predict(map_rdm[(nbmodels-nbtest):,:,:])

#Prediction on random model with random and smooth
pred_rdm = model_rdm.predict(mymaps[(nbmodels-nbtest):,:,:])
pred_rdm2 = model_rdm.predict(map_rdm[(nbmodels-nbtest):,:,:])

#Saves
dump(mymodel, out_dir + "/mymodel.joblib")
np.save(out_dir + "/prediction", prediction)
np.save(out_dir + "/prediction2", prediction2)
np.save(out_dir + "/pred_rdm", pred_rdm)
np.save(out_dir + "/pred_rdm2", pred_rdm2)
np.save(out_dir + "/loss-alm", hist.history['loss'])
np.save(out_dir + "/val_loss-alm", hist.history['val_loss'])
np.save(out_dir + "/loss-alm_rdm", hist_rdm.history['loss'])
np.save(out_dir + "/val_loss-alm_rdm", hist_rdm.history['val_loss'])