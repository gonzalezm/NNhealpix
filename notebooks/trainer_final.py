import numpy as np
import keras as kr
import datetime
import healpy as hp
import nnhealpix
import nnhealpix.layers
import sys
import os
from ConvNNTempLib import *

# Directory selection
dir = sys.argv[1]
in1 = sys.argv[2] #Maps and lp or Cl
in2 = sys.argv[3] #model and history
name = sys.argv[4] #Maps and lp or Cl

date = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
out_dir = dir + '{}/'.format(date) 

#Take data
Maps = np.load(dir + in1 + "/" + name + "_Maps.npy")
C_l = np.load(dir + in1 + "/" + name + "_C_l.npy")

with kr.utils.CustomObjectScope({'OrderMap':nnhealpix.layers.OrderMap}):
    model=kr.models.load_model(dir + in2 + "/" + in2 + "_model.h5py.File")

Ntest = 0.1
Ntrain = 1-Ntest
sigma_n = 0.
Nside = hp.npix2nside(Maps.shape[1])

Nmodel = C_l.shape[0]
#Ntest = int(Ntest*Nmodel)
#Ntrain = int(Ntrain*Nmodel)

Maps = AddWhiteNoise(Maps,sigma_n)
Maps = NormalizeMaps(Maps)

X_train, y_train, X_test, y_test, num_out, shape = PreprocessML(Maps, C_l, Ntest, Ntrain)

#Creat the repository where save new data
try:
    os.makedirs(out_dir)
except:
    pass

model, hist, loss2, val_loss2 = MakeAndTrainModel(X_train, y_train,
                                                X_test, y_test,
                                                epoch = 10, batch_size = 32,
                                                out_dir = out_dir, today = date,
                                                retrain=True, model=model)

error = model.evaluate(X_test, y_test)
print('error :', error)

loss1 = np.load(dir + in2 + "/" + in2 + '_hist_loss.npy')
val_loss1 = np.load(dir + in2 + "/" + in2 + '_hist_val_loss.npy')

loss = np.concatenate((loss1,loss2))
val_loss = np.concatenate((val_loss1,val_loss2))

# Save the model as a pickle in a file
kr.models.save_model(model, out_dir + date + '_model.h5py.File')

#np.save(out_dir + today + '_prediction', prediction)
np.save(out_dir + date + '_hist_loss', loss)
np.save(out_dir + date + '_hist_val_loss', val_loss)
np.save(out_dir + date + '_history', hist.history)