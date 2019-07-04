import camb
from camb import model, initialpower
import numpy as np
import healpy as hp
from pylab import *
import sys
import os
import datetime

# Directory where files will be saved

dir = sys.argv[1]
now = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
out_dir = dir + '{}/'.format(now)

try:
    os.makedirs(out_dir)
except:
    pass

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

totCL = powers['total']
unlensedCL = powers['unlensed_scalar']
print(totCL.shape)

ls = np.arange(totCL.shape[0])
CL = totCL[:, 0]/ls/(ls+1)*2*np.pi
CL[0]=0

#some parameters
ns = 8
lmax = 2*ns-1
nl = 2*ns
nalm = (2*ns)*(2*ns+1)/2

#Data creation
map =hp.synfast(CL[0:5*ns], ns, pixwin=False, verbose= False)
outcl = hp.anafast(map, alm=False, lmax = lmax)
ll = np.arange(outcl.shape[0])
lt = ll[0:nl]

nbmodels = sys.argv[2]
nbtest = int(0.1*nbmodels)
nnn = int(nbmodels/30)
npixok = 12*ns**2
limit_shape = 3*ns
okpix = np.arange(npixok)
mymaps = np.zeros((nbmodels, npixok))
myalms = np.zeros((nbmodels, int(nalm)), dtype = complex128)
expcls = np.zeros((nbmodels, nl))
mycls = np.zeros((nbmodels, nl))
allshapes = np.zeros((nbmodels, len(ls)))
#Create random map and random spectrum from no constant or theorical model  
cls_rdm = np.random.rand(nbmodels, mycls.shape[1])*(CL.max())
map_rdm = np.zeros((nbmodels, len(mymaps[0])))
ana_rdm = np.zeros((cls_rdm.shape[0], cls_rdm.shape[1]))
for i in range(nbmodels):
    ylo = np.random.rand()*2
    yhi = np.random.rand()*2
    theshape = ylo+(yhi-ylo)/(limit_shape)*ls
    theshape[theshape < 0] = 0
    theshape[limit_shape:] = 0
    theCL = CL*theshape
    themap = hp.synfast(theCL, ns, pixwin=False, verbose = False)
    mymaps[i,:] = themap[okpix]
    expcls[i,:], myalms[i,:] = hp.anafast(themap, lmax = lmax, alm = True)
    mycls[i,:] = theCL[0:nl]
    map_rdm[i,:] = hp.synfast(cls_rdm[i,:], ns, pixwin=False, verbose = False)
    ana_rdm[i,:] = hp.anafast(map_rdm[i,:], lmax = lmax, alm = False)

#All the saves
np.save(out_dir + "/lt", lt )
#np.save(out_dir + "/myalms", myalms )
np.save(out_dir + "/mycls", mycls )
np.save(out_dir + "/expcls", expcls )
np.save(out_dir + "/mymaps", mymaps )
np.save(out_dir + "/cls_rdm", cls_rdm )
np.save(out_dir + "/map_rdm", map_rdm )
np.save(out_dir + "/ana_rdm", ana_rdm )