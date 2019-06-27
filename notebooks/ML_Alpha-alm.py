import camb
from camb import model, initialpower
import tensorflow as tf
import math
import keras as kr
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from keras.models import Sequential
import nnhealpix as nn
import nnhealpix.layers
from pylab import *
import sys
import os
import datetime

# Directory where files will be saved
#
#dir = sys.argv[1]
#today = datetime.datetime.now().strftime('%Y%m%d')
#out_dir = dir + '{}/'.format(today)
#
#try:
#    os.makedirs(out_dir)
#except:
#    pass

rcParams['image.cmap'] = 'jet'

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(ns=0.965, r=0)
pars.set_for_lmax(2500, lens_potential_accuracy=0);
#calculate results for these parameters
results = camb.get_results(pars)
#get dictionary of CAMB power spectra
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')

#plot the total lensed CMB power spectra versus unlensed, and fractional difference
totCL = powers['total']
unlensedCL = powers['unlensed_scalar']
print(totCL.shape)

ls = np.arange(totCL.shape[0])
CL=np.zeros(totCL.shape[0])
CL[1:] = totCL[1:,0]/ls[1:]/(ls[1:]+1)


#some parameters
ns = 16
lmax = 2*ns-1
nl = 2*ns
nalm = (2*ns)*(2*ns+1)/2

#Data creation
map =hp.synfast(CL[0:5*ns], ns, pixwin=False, verbose= False)
outcl = hp.anafast(map,alm=False, lmax=lmax)
# /!\ anafast gives us cl and alm and we need to take cl only, reason why we take anafast(...)[0]
ll = np.arange(outcl.shape[0])
lt = ll[0:nl]

nbmodels = np.int(sys.argv[2])
nbtest = int(0.1*nbmodels)
nnn = int(nbmodels/30)
npixok = 12*ns**2
limit_shape = 3*ns
okpix = np.arange(npixok)
mymaps = np.zeros((nbmodels, npixok))
expcls = np.zeros((nbmodels, nl))
mycls = np.zeros((nbmodels, nl))
expcls = np.zeros((nbmodels, nl))
allshapes = np.zeros((nbmodels, len(ls)))
for i in range(nbmodels):
  ylo = np.random.rand()*2
  yhi = np.random.rand()*2
  theshape = ylo+(yhi-ylo)/(limit_shape)*ls
  theshape[theshape < 0] = 0
  theshape[limit_shape:] = 0
  allshapes[i,:] = theshape
  theCL = CL*theshape
  themap = hp.synfast(theCL, ns, pixwin=False, verbose = False)
  mymaps[i,:] = themap[okpix]
  #expcls find with anafast, 
  expcls[i,:] = hp.anafast(themap, lmax=lmax, alm = False)
  mycls[i,:] = theCL[0:nl]
  
#Machine learning
#Data preprocess  

class PrintNum(kr.callbacks.Callback):
  def on_epoch_end(self,epoch,logs):
    if epoch % 10 == 0: 
      print('')
      print(epoch, end='')
    sys.stdout.write('.')
    sys.stdout.flush()

print("La shape de mymaps (inputs): ", mymaps.shape)
print("La shape de mycls (outputs): ", mycls.shape)
shape=(len(mymaps[0,:]),1)
mymaps = mymaps.reshape(mymaps.shape[0], len(mymaps[0]), 1)
myclsmod = lt*(lt+1)*mycls/(2*np.pi)

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
hist = mymodel.fit(mymaps[:(nbmodels-nbtest),:,:], myclsmod[:(nbmodels-nbtest),:], epochs=5, batch_size=32, validation_split = 0.1, verbose = 1, shuffle = True)

#Make a prédiction
prediction=mymodel.predict(mymaps[(nbmodels-nbtest):,:,:])

np.save(out_dir + "/loss-alm", hist.history['loss'])
np.save(out_dir + "/val_loss-alm", hist.history['val_loss'])
np.save(out_dir + "/mycls", mycls )
np.save(out_dir + "/myclsmod", myclsmod )
np.save(out_dir + "/expcls", expcls )
np.save(out_dir + "/mymaps", mymaps )
np.save(out_dir + "/lt", lt )