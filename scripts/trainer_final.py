import numpy as np
import keras as kr
import datetime
import healpy as hp
import nnhealpix.layers
import sys
import os

import ConvNNTempLib as cnn

# Directory selection
name = sys.argv[1]  # Maps and lp or Cl
in1 = sys.argv[2]  # Maps and lp or Cl
in2 = sys.argv[3]  # model and history
dir = sys.argv[4] # Directory path for saving results


date = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
out_dir = dir + '{}/'.format(date)

# Creat the repository where save new data
os.makedirs(out_dir, exist_ok=True)

# Take data
maps = np.load(dir + in1 + "/" + name + "_maps.npy")
C_l = np.load(dir + in1 + "/" + name + "_C_l.npy")

# Load a model already trained
with kr.utils.CustomObjectScope({'OrderMap': nnhealpix.layers.OrderMap}):
    model = kr.models.load_model(dir + in2 + "/" + in2 + "_model.h5py.File")

# Get Nside
nside = hp.npix2nside(maps.shape[1])

# Add noise and normalize the maps
sigma_n = 0.
maps = cnn.AddWhiteNoise(maps, sigma_n)
maps = cnn.NormalizeMaps(maps)

Nmodel = C_l.shape[0]
# Ntest = int(Ntest*Nmodel)
# Ntrain = int(Ntrain*Nmodel)

ntest = 0.1
ntrain = 1 - ntest
x_train, y_train, x_test, y_test, num_out, shape = cnn.PreprocessML(maps, C_l, ntest, ntrain)

model, hist, loss2, val_loss2 = cnn.MakeAndTrainModel(x_train, y_train,
                                                  x_test, y_test,
                                                  epoch=10, batch_size=32,
                                                  out_dir=out_dir, today=date,
                                                  retrain=True, model=model)

error = model.evaluate(x_test, y_test)
print('error :', error)

loss1 = np.load(dir + in2 + "/" + in2 + '_hist_loss.npy')
val_loss1 = np.load(dir + in2 + "/" + in2 + '_hist_val_loss.npy')

loss = np.concatenate((loss1, loss2))
val_loss = np.concatenate((val_loss1, val_loss2))

# Save the model as a pickle in a file
kr.models.save_model(model, out_dir + date + '_model.h5py.File')

# np.save(out_dir + today + '_prediction', prediction)
np.save(out_dir + date + '_hist_loss', loss)
np.save(out_dir + date + '_hist_val_loss', val_loss)
np.save(out_dir + date + '_history', hist.history)
