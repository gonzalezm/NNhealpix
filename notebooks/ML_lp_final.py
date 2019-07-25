import keras as kr
import healpy as hp
import numpy as np
import sys
import os
import datetime
from ConvNNTempLib import AddWhiteNoise, NormalizeMaps, PreprocessML, ConvNNhealpix, MakeAndTrainModel

#Take in arguments path where find the data and path where save new data.
#Take the name given during the creation of data
name_in = sys.argv[1]
in_dir = sys.argv[2]
out_dir = sys.argv[3]

#Add date and time to the path to save to avoid "same name file" problems.
today = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
out_dir += '/{}/'.format(today)

# Load the data
l_p = np.load(in_dir + "/" + name_in + "_l_p.npy")
Maps = np.load(in_dir + "/" + name_in + "_Maps.npy")

Ntest = 0.1
Ntrain = 1-Ntest
sigma_n = sys.argv[4]
Nside = hp.npix2nside(Maps.shape[1])

Maps = AddWhiteNoise(Maps, sigma_n)
Maps = NormalizeMaps(Maps)
X_train, y_train, X_test, y_test, num_out, shape = PreprocessML(Maps, l_p, Ntest, Ntrain)

inputs, out = ConvNNhealpix(shape, Nside, num_out)


#Creat the repository where save new data
try:
    os.makedirs(out_dir)
except:
    pass

model, hist, loss, val_loss = MakeAndTrainModel(X_train, y_train,
                                                X_test, y_test,
                                                epoch = 20, batch_size = 32,
                                                out_dir = out_dir, today = today,
                                                inputs = inputs, out = out,
                                                retrain=False)

error = model.evaluate(X_test, y_test)
print('error :', error)

# Save the model as a pickle in a file
kr.models.save_model(model, out_dir + today + '_model.h5py.File')

#np.save(out_dir + today + '_prediction', prediction)
np.save(out_dir + today + '_hist_loss', hist.history['loss'])
np.save(out_dir + today + '_hist_val_loss', hist.history['val_loss'])
np.save(out_dir + today + '_history', hist.history)