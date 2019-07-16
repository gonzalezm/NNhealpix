import keras as kr
import healpy as hp
import nnhealpix
import numpy as np
import math
import camb
import scipy.stats as stats

def CreateGaussianMapsAndCl(Nmodel, sigma_p, Nside):
    """
    Create Gaussian spectra and Healpix Maps with them.
    ========= Parameters ========================
    Nmodel: An integer, the number of C_l, l_p and Maps needed
    
    sigma: The standard deviation of the gaussian
    
    Nside: An integer and power of 2, Give the resolution of the final maps
    
    ========= Return ==========================
    l_p: A list of float, the gaussians means
    
    C_l: A 2D array of float, the gaussian points
    
    Maps: Healpix maps of the gaussian spectra
    """
    lp_min=5.0
    lp_max= 2.0*Nside
    l_p = np.random.random_sample(lp_min, lp_max, Nmodel)
    
    l = np.arange(2*lp_max)
    C_l = np.empty((len(l), Nmodel))
    Maps = np.empty((12 * Nside ** 2, Nmodel))
    
    for j in range (len(l_p)):
        C_l[:, j] = stats.norm.pdf(l, l_p[j], sigma_p) + 10.**(-5)
        Maps[:, j] = hp.sphtfunc.synfast(C_l[:, j], Nside, verbose = 0)
    
    return l_p, C_l, Maps

def CreateRealisticData(Nmodels, Nside):
    """
    Create a set of data with realistic maps of CMB, spectra and alm
    ============= Parameters ===================================
    Nmodels: An integer, the number of model.
    
    Nside: The resolution of the maps created
    
    ================ Return ====================================
    mymaps: A 2D array of float, realistic CMB maps.
    
    mycls: A 2D array of float, realistic spectra.
    
    expcls: A 2D array of float, spectra obtained by anafast.
    
    myalms: An array of complex obtained by anafast.
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(ns=0.965, r=0)
    pars.set_for_lmax(2500, lens_potential_accuracy=0);
    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
    
    totCL=powers['total']
    
    ls = np.arange(totCL.shape[0])
    CL = totCL[:,0]/ls/(ls+1)
    CL[0]=0
    
    lmax = 2*Nside-1
    nl = 2*Nside
    nalm = (nl)*(nl+1)/2
    npixok = 12*Nside**2
    limit_shape = 3*Nside
    okpix = np.arange(npixok)
    mymaps = np.zeros((Nmodels, npixok))
    myalms = np.zeros((Nmodels, int(nalm)), dtype=complex128)
    expcls = np.zeros((Nmodels, nl))
    mycls = np.zeros((Nmodels, nl))
    allshapes = np.zeros((Nmodels, len(ls)))
    for i in range(Nmodels):
        ylo = np.random.rand()*2
        yhi = np.random.rand()*2
        theshape = ylo+(yhi-ylo)/(limit_shape)*ls
        theshape[theshape < 0] = 0
        theshape[limit_shape:] = 0
        allshapes[i,:] = theshape
        theCL = CL*theshape
        themap = hp.synfast(theCL, Nside, pixwin=False, verbose=False)
        mymaps[i,:] = themap[okpix]
        expcls[i,:], myalms[i,:] = hp.anafast(themap, lmax=lmax, alm=True)
        mycls[i,:] = theCL[0:nl]

    return mymaps, mycls, expcls, myalms

def AddWhiteNoise(inp,sigma_n):
    """
    Add a gaussian white noise on the map
    ============ Parameters =================
    inp: A 2D array of float, the map which will have noise added
    
    sigma_n: A float, the standard deviation of the gaussian noise
    
    ============ Return ====================
    inp_n: A 2D array, maps with white noise
    """
    inp_n = inp + np.random.randn(inp.shape[0],inp.shape[1])*sigma_n
    return inp_n

def NormalizeMaps(Maps):
    """
    Normalize the maps
    ============ Parameters ================
    Maps: An array of float
    ============ Return ====================
    MapsNorm: An array of float normalized
    """
    MapsNorm = (Maps-np.mean(Maps))/np.std(Maps)
    return MapsNorm

def PreprocessML(inp, out, Ntest, Ntrain, sigma_n):
    """
    Prepare the data for the ML with ND inputs and outputs
    N>2
    ========= Parameters ===============
    inp:A 2D array of float, The last axis is for separate the different data.
        The input of the machine learning
        
    out: An 1D array of float. The last axis give de i-th data.
    /!\ inp.shape[1] must be equal to out.shape[0]
    
    Ntest: An integer or a float between 0 and 1. The number or quantity of validation input and outputs.
    
    Ntest: An integer or a float between 0 and 1. The number or quantity of training input and outputs.
    
    /!\ if float: Ntest+Ntrain<=1
    /!\ if integer: Ntest+Ntrain<= Maps.shape[1]
    
    sigma_n: A float or an integer, the standard deviation of the gaussian white noise.
    ========= Return ==================
    X_train: A 3D array of float, the training input data
    
    y_train: A 1 or 2D array or list of float, the training output data
    
    X_test: A 3D array of float, the validation input data
    
    y_test: A 1 or 2D array of float, the validation output data
    
    num_out: integer, the number of neuron of the output layer, it depends on the situation.
    
    shape: a turple with the shape of ONE input map.
    """
    
    Nmodel = out.shape[0]
    if Ntest != int(Ntest) and Ntest >= 0.0 and Ntest <= 1.0:
        Ntest = int(Ntest*Nmodel)
    if Ntrain != int(Ntrain) and Ntrain >= 0.0 and Ntrain <= 1.0:
        Ntrain = int(Ntrain*Nmodel)
    
    inp = AddWhiteNoise(inp, sigma_n)
    if len(out.shape) == 1:
        num_out=1
    else:
        out = out.T
        num_out = out.shape[1]
        
    # Split between train and test
    inp = NormalizeMaps(inp)
    X_train = inp[:,:Ntrain]
    y_train = out[0:Ntrain,:]
    X_test = inp[:,Ntrain:(Ntrain+Ntest)]
    y_test = out[Ntrain:(Ntrain+Ntest),:]
    
    X_train = X_train.T
    X_test = X_test.T
    X_train = X_train.reshape(X_train.shape[0], len(X_train[0]), 1)
    X_test = X_test.reshape(X_test.shape[0], len(X_test[0]), 1)
    
    shape = (len(inp[:, 0]), 1)
    
    return X_train, y_train, X_test, y_test, num_out, shape

def PreprocessNDimML(inp, out, Ntest, Ntrain):
    """
    Prepare the data for the ML with ND inputs and outputs
    N>2
    ========= Parameters ===============
    inp:A ND array of float, The last axis is for separate the different data.
        The input of the machine learning
        
    out: An ND array of float. The last axis give de i-th l_p,C_l or other.
    /!\ inp.shape[1] must be equal to out.shape[0]
    
    Ntest: An integer or a float between 0 and 1. The number or quantity of validation input and outputs
    
    Ntest: An integer or a float between 0 and 1. The number or quantity of training input and outputs
    
    /!\ if float: Ntest+Ntrain<=1
    /!\ if integer: Ntest+Ntrain<= Maps.shape[1]
    ========= Return ==================
    X_train: A ND array of float, the training input data
    
    y_train: A ND array or list of float, the training output data
    
    X_test: A ND array of float, the validation input data
    
    y_test: A ND array of float, the validation output data
    
    num_out: integer, the number of neuron of the output layer, it depends on the situation.
    
    shape: a turple with the shape of ONE input map.
    """
    
    Nmodel = inp.shape[len(inp.shape)-1]
    if Ntest != int(Ntest) and Ntest >= 0.0 and Ntest <= 1.0:
        Ntest = int(Ntest*Nmodel)
    if Ntrain != int(Ntrain) and Ntrain >= 0.0 and Ntrain <= 1.0:
        Ntrain = int(Ntrain*Nmodel)
    
    num_out = 0
    for i in range (len(out.shape)-1):
        num_out += out.shape[i]
    
    inp=np.moveaxis(inp, len(inp.shape)-1, 0)
    out=np.moveaxis(out, len(out.shape)-1, 0)
    inp = NormalizeMaps(inp)
    # Split between train and test
    mod_range=np.arange(Nmodel)
    X_train = np.compress(mod_range[:Ntrain],inp,axis=0)
    y_train = np.compress(mod_range[0:Ntrain],out,axis=0)
    X_test = np.compress(mod_range[Ntrain:(Ntrain+Ntest)],inp,axis=0)
    y_test = np.compress(mod_range[Ntrain:(Ntrain+Ntest)],out,axis=0)
    
    shape = X_train.shape[1:]
    
    return X_train, y_train, X_test, y_test, num_out, shape


def ConvNNhealpix(shape, nside, num_out):
    """
    Architecture of the Neural Network using the NNhealpix functions
    
    ========= Parameters =============
    shape: a turple with the shape of ONE input map.
    
    nside: integer with the resolution parameter of your input maps, must be a power of 2.
    
    num_out: integer, the number of neuron of the output layer, it depends on the situation.
    
    ========= Return ================
    out: the last layer of the network of num_out neurons
    """
    inputs = kr.layers.Input(shape)
    x = inputs
    
    # NBB loop (conv and degrade from nside to 1)
    
    for i in range(int(math.log(nside, 2))):
        # Recog of the neighbours & Convolution
        print(int(nside / (2 ** i)), int(nside / (2 ** (i + 1))))
        x = nnhealpix.layers.ConvNeighbours(int(nside / (2 ** i)),
                                            filters=32,
                                            kernel_size=9)(x)
        x = kr.layers.Activation('relu')(x)
        # Degrade
        x = nnhealpix.layers.MaxPooling(int(nside / (2 ** i)),
                                        int(nside / (2 ** (i + 1))))(x)
    
    # End of the NBBs
    
    x = kr.layers.Dropout(0.2)(x)
    x = kr.layers.Flatten()(x)
    x = kr.layers.Dense(48)(x)
    x = kr.layers.Activation('relu')(x)
    x = kr.layers.Dense(num_out)(x)
    
    out = kr.layers.Activation('relu')(x)
    
    return out