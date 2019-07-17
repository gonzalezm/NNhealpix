import keras as kr
import healpy as hp
import nnhealpix
import nnhealpix.layers
import numpy as np
import math
import camb
import matplotlib.pyplot as plt
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
    pars.set_for_lmax(2500, lens_potential_accuracy=0)
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

def PreprocessML(inp, outp, Ntest, Ntrain, sigma_n):
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
    
    Nmodel = outp.shape[0]
    if Ntest != int(Ntest) and Ntest >= 0.0 and Ntest <= 1.0:
        Ntest = int(Ntest*Nmodel)
        print("Ntest={}".format(Ntest))
    if Ntrain != int(Ntrain) and Ntrain >= 0.0 and Ntrain <= 1.0:
        Ntrain = int(Ntrain*Nmodel)
        print("Ntest={}".format(Ntest))
    
    inp = AddWhiteNoise(inp, sigma_n)
    if len(outp.shape) == 1:
        num_out=1
        outp = outp.reshape(outp.shape[0], 1)
    else:
        outp = outp.T
        num_out = outp.shape[1]
        
    # Split between train and test
    inp = NormalizeMaps(inp)
    X_train = inp[:,:Ntrain]
    y_train = outp[0:Ntrain,:]
    X_test = inp[:,Ntrain:(Ntrain+Ntest)]
    y_test = outp[Ntrain:(Ntrain+Ntest),:]
    
    X_train = X_train.T
    X_test = X_test.T
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
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
    inputs: The inputs for the first layer of the Network.
    
    out: the last layer of the network of num_out neurons.
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
    
    return inputs, out

def MakeAndTrainModel(inputs, out, X_train, y_train, X_test, y_test, epoch, batch_size, out_dir, today=None):
    """
    Create a model from a Network, show this network and train the model.
    =================== Parameters ================================
    inputs:
    
    out:
        
    Xtrain:
        
    y_train:
        
    X_test:
        
    y_test:
        
    epoch:
        
    batch_size:
        
    out_dir,
    
    today
    =================== Results ===================================
    
    """
    model = kr.models.Model(inputs=inputs, outputs=out)
    model.compile(loss=kr.losses.mse,
                  optimizer='adam',
                  metrics=[kr.metrics.mean_absolute_percentage_error])
    model.summary()
    
    # Callbacks
    checkpointer_mse = kr.callbacks.ModelCheckpoint(filepath=out_dir + today + '_weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                    monitor='val_loss',
                                                    verbose=1,
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    mode='min',
                                                    period=1)
    
    stop = kr.callbacks.EarlyStopping(monitor='val_loss',
                                      verbose = 0,
                                      restore_best_weights = True,
                                      patience=20)
    
    callbacks = [checkpointer_mse, stop]
    
    # Model training
    # model._ckpt_saved_epoch = None
    hist = model.fit(X_train, y_train,
                     epochs=20,
                     batch_size=32,
                     validation_data=(X_test, y_test),
                     verbose=1,
                     callbacks=callbacks,
                     shuffle=True)
    
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    
    return model, hist, loss, val_loss

def PlotLosses(loss, val_loss):
    """
    Show the loss and the validation loss of a model training
    ========= Parameters =======================
    hist: An array with the data of the training
    
    ========= Return ===========================
    Show the loss and the validation loss
    """
    
    fig = plt.figure(1, figsize=(10,10))
    plt.plot(loss, ':', color = 'blue', label = "loss")
    plt.plot(val_loss, '--', color = 'green', label = "val_loss")
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title("Losses and val_losses as function of epochs")
    plt.legend(loc=1)
    plt.yscale('log')
    plt.show()
    return

def MakePrediction(inp, model, sigma_n=0):
    """
    Make a prediction with given model and data, add also nois to the data
    ============ Parameters =============================
    inp: An array of float. The input data.
    
    model: The given model
    
    sigma_n: A float or n integer, the standard deviation of the gaussian white noise.
    ============ Return =================================
    pred: An array or a list with the predicted data.
    """
    inp = AddWhiteNoise(inp, sigma_n)
    inp = NormalizeMaps(inp)
    inp = inp.reshape(inp.shape[0], inp.shape[1], 1)
    pred = model.predict((inp))
    return pred

def PlotError(pred, y_test):
    """
    Calculate and plot the histogram of the error
     ================= Parameters ==================
    pred: An array with the data predict by a model
    
    y_test: An array with the expected data
    /!\ y_test and pred must have the same shape, in a perfect case y_test == pred
    ================= Return ======================
    
    Show the plot of the error histogram
    """
    err = (pred-y_test)/y_test*100
    mean_err = np.mean(abs(err))
    print('Mean relative error: ', mean_err)
    if len(pred.shape) != 1 or pred.shape[1:] != 1:
        err=np.ravel(err)
    fig = plt.figure(1, figsize=(10,10))
    plt.hist(err, bins=100, alpha=0.5, label='err', color = 'blue')
    plt.title("Histogram of the error between the predicted and expected data")
    plt.xlabel("Difference between the predicted and expected data")
    plt.ylabel("Amount of points")
    plt.legend(loc=1)
    plt.show()
    return
    

def PlotChi2(pred, y_test):
    """
    Calculate and plot the histogram of the chi2. 
    ================= Parameters ==================
    pred: An array with the data predict by a model
    
    y_test: An array with the expected data
    /!\ y_test and pred must have the same shape, in a perfect case y_test == pred
    ================= Return ======================
    
    Show the plot of the chi2 histogram
    """
    chi2 = np.sum((pred-y_test)**2, axis = 1)
    fig= plt.figure(1, figsize=(10,10))
    plt.hist(chi2, color = "blue", bins=100, alpha=0.5, label='$\chi²$')
    plt.yscale('log')
    plt.title("$\chi^2$")
    plt.legend(loc=1)
    plt.show()
    return

def PlotInOnOut1D(pred, y_test):
    """
    ================= Parameters ==================
    pred: A 1D array or a list with the data predict by a model
    
    y_test: A 1D array or a list with the expected data
    ================= Return ======================
    Show the plot of the prediction as a function of the test data
    """
    fig= plt.figure(1, figsize=(10,10))
    plt.plot(y_test,pred,' ', marker='.', label= "20ep", color = 'blue')
    #plt.plot(ytest,pred2,' ', marker='.', label= "30ep", color = 'red')
    plt.plot(np.linspace(min(y_test),max(y_test),int(max(y_test)-min(y_test))+1), np.linspace(min(y_test),max(y_test),int(max(y_test)-min(y_test))+1), lw=2, label = "Expected", color = 'green')
    plt.xlabel("Test data")
    plt.ylabel("Predicted data")
    plt.legend(loc=4)
    plt.show()
    return