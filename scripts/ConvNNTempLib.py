import keras as kr
import healpy as hp
import nnhealpix
import nnhealpix.layers
import numpy as np
import math
import camb
import matplotlib.pyplot as plt


def CreateGaussianMapsAndCl(nmodel, sigma_p, nside):
    """
    Create Gaussian spectra and associated Healpix maps.
    Return also the means of the gaussian.

    ========= Parameters ========================
    nmodel: int
        Number of C_l, l_p and Maps you want to create.
    sigma: float
        Gaussian standard deviations.
    nside: int
        Resolution for the maps, power of 2.
    
    ========= Return ==========================
    lp: list, len = 4 x nside
        Gaussian means random between 5 and 2xnside
    cl: 2D array
        Set of gaussian spectra, shape (nmodel, 4 x nside)
    maps: 2D array
        Set of maps of the full sky, shape (nmodel, #pixels)
    """

    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) + 1e-5

    l = np.arange(4 * nside)

    maps = np.empty((nmodel, hp.nside2npix(nside)))
    cl = np.empty((nmodel, len(l)))
    lp = np.empty(nmodel)

    for i in range(nmodel):
        np.random.seed()
        lp[i] = np.random.uniform(5., 2. * nside)
        cl[i, :] = gaussian(l, lp[i], sigma_p)
        maps[i, :] = hp.synfast(cl[i, :], nside, verbose=0)

    return lp, cl, maps


def CreateRealisticData(nmodels, nside):
    """
    Create a set of data with realistic maps of CMB, spectra and alm

    ============= Parameters ===================================
    nmodels: int
        Number of models.
    nside: int
        The resolution of the maps created

    ================ Return ====================================
    mymaps: 2D array
        Set of maps of the full sky, first indice is the number of the map
        second indice is the number of the pixel on the map, realistic CMB maps.
    mycls: 2D array
        Set of realistic spectra, first indice is the number of the spectrum
        second indice is the value of the spectrum.
    expcls: 2D array
        Set of anafast spectra, first indice is the number of the spectrum
        second indice is the value of the spectrum.
    myalms: complex array
        Spectrum computed with anafast.
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(ns=0.965, r=0)
    pars.set_for_lmax(2500, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')

    totCL = powers['total']

    ls = np.arange(totCL.shape[0])
    CL = totCL[:, 0] / ls / (ls + 1)
    CL[0] = 0

    lmax = 2 * nside - 1
    nl = 2 * nside
    nalm = nl * (nl + 1) / 2
    npixok = 12 * nside ** 2
    limit_shape = 3 * nside
    okpix = np.arange(npixok)
    mymaps = np.zeros((nmodels, npixok))
    myalms = np.zeros((nmodels, int(nalm)), dtype=np.complex128)
    expcls = np.zeros((nmodels, nl))
    mycls = np.zeros((nmodels, nl))
    allshapes = np.zeros((nmodels, len(ls)))
    for i in range(nmodels):
        ylo = np.random.rand() * 2
        yhi = np.random.rand() * 2
        theshape = ylo + (yhi - ylo) / limit_shape * ls
        theshape[theshape < 0] = 0
        theshape[limit_shape:] = 0
        allshapes[i, :] = theshape
        theCL = CL * theshape
        themap = hp.synfast(theCL, nside, pixwin=False, verbose=False)
        mymaps[i, :] = themap[okpix]
        expcls[i, :], myalms[i, :] = hp.anafast(themap, lmax=lmax, alm=True)
        mycls[i, :] = theCL[0:nl]

    return mymaps, mycls, expcls, myalms


def AddWhiteNoise(inp, sigma_n):
    """
    Add a gaussian white noise on the map
    ============ Parameters =================
    inp: A 2D array, the set of maps of the full sky, first indice is the number of the map
        second indice is the number of the pixel on the map
    sigma_n: A float, the standard deviation of the gaussian noise
    
    ============ Return ====================
    inp_n: A 2D array, the set of maps of the full sky, first indice is the number of the map
        second indice is the number of the pixel on the map, maps with white noise
    """
    inp_n = inp + np.random.randn(inp.shape[0], inp.shape[1]) * sigma_n
    return inp_n


def NormalizeMaps(map):
    """
    Normalize a map
    """
    return (map - np.mean(map)) / np.std(map)


def MakePatchMaps(maps, theta, phi, radius):
    """
    Transform a set of maps of the full sky in a set of map of a sky patch

    ============ Parameters ==============================
    Maps: 2D array
        Set of maps of the full sky, first indice is the number of the map
        second indice is the number of the pixel on the map.
    theta: float,
        Angle in radian for the center of the patch
    phi: float
        Angle in radian for the center of the patch
    radius: float
        Radius of the patch in radian

    ============ Return ==================================
    map_patch: A 2D array, the set of maps of a sky patch, first indice is the number of the map
        second indice is the number of the pixel on the map.
    """
    vec = hp.ang2vec(theta, phi)

    nside = hp.npix2nside(maps.shape[1])
    # make the map with only the patch from full maps
    patch = hp.query_disc(nside, vec, radius)
    map_patch = np.full((maps.shape[0], maps.shape[1]), hp.UNSEEN)
    for i in range(maps.shape[0]):
        map_patch[i, patch] = maps[i, patch]

    return map_patch


def PreprocessML(inp, outp, ntest, ntrain, patch=False, theta=0, phi=0, r=0):
    """
    Prepare the data for the ML with ND inputs and outputs

    ========= Parameters ===============
    inp:2D array
        Set of maps of the full sky, first index is the number of the map
        second index is the number of the pixel on the map.
        The input of the machine learning
    outp: 1 or 2-D array
          inp.shape[0] must be equal to outp.shape[0]
    ntest: float
        Fraction for validation data between 0 and 1.
    sigma_n: float
        She standard deviation of the gaussian white noise.
    patch: bool
        Say if you want a patch of the sky (True) or the full sky (False)
    theta: float,
        Angle in radian for the center of the patch
    phi: float
        Angle in radian for the center of the patch
    radius: float
        Radius of the patch in radian

    ========= Return ==================
    x_train: A 3D array of float, the training input data
    y_train: A 1 or 2D array or list of float, the training output data
    x_test: A 3D array of float, the validation input data
    y_test: A 1 or 2D array of float, Validation output data
    num_out: int
        Number of neurons of the output layer.
    shape: tuple
        Shape of ONE input map.
    """

    nmodel = outp.shape[0]
    if ntest != int(ntest) and 0.0 <= ntest <= 1.0:
        ntest = int(ntest * nmodel)
        print("ntest={}".format(ntest))
    if ntrain != int(ntrain) and 0.0 <= ntrain <= 1.0:
        ntrain = int(ntrain * nmodel)
        print("ntrain={}".format(ntrain))

    if len(outp.shape) == 1:
        outp = outp.reshape(outp.shape[0], 1)
    num_out = outp.shape[1]

    if patch:
        inp = MakePatchMaps(inp, theta, phi, r)

    # Split between train and test
    x_train = inp[:ntrain, :]
    y_train = outp[0:ntrain, :]
    x_test = inp[ntrain:(ntrain + ntest), :]
    y_test = outp[ntrain:(ntrain + ntest), :]

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    shape = (inp.shape[1], 1)

    return x_train, y_train, x_test, y_test, num_out, shape


def ConvNNhealpix(shape, nside, num_out):
    """
    Architecture of the Neural Network using the NNhealpix functions
    
    ========= Parameters =============
    shape: tuple
        Shape of ONE input map.
    nside: int
        Resolution parameter of your input maps, must be a power of 2.
    num_out: int
        Number of neuron of the output layer.
    
    ========= Return ================
    inputs: keras tensor
        Input used to build the network.
    out: keras tensor
        Last layer of the network.
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
        x = nnhealpix.layers.AveragePooling(int(nside / (2 ** i)),
                                            int(nside / (2 ** (i + 1))))(x)

    # End of the NBBs

    x = kr.layers.Dropout(0.2)(x)
    x = kr.layers.Flatten()(x)
    x = kr.layers.Dense(256)(x)
    x = kr.layers.Activation('relu')(x)
    x = kr.layers.Dense(num_out)(x)

    out = kr.layers.Activation('relu')(x)

    return inputs, out


def MakeAndTrainModel(x_train, y_train,
                      x_test, y_test,
                      epoch, batch_size,
                      out_dir, today=None,
                      inputs=None, out=None,
                      retrain=False, preexisting_model=None):
    """
    Create a model from a Network, show this network and train the model.

    =================== Parameters ================================
    x_train: 3D array of float
        Training input data.
    y_train: A 1 or 2D array or list of float
        Training output data.
    x_test: A 3D array of float
        Validation input data.
    y_test: A 1 or 2D array of float
        Validation output data.
    epoch: int
        Number of epochs.
    batch_size: int
        Batch size.
    out_dir: str
        Repository of the saved weights.
    today: str
        Name for the weights that will be saved.
    inputs: keras tensor
        Input used to build the network if retrain=False.
    out: keras tensor
        Last layer of the network if retrain=False.
    retrain: bool
        Set True if you want to load a model already trained.
    preexisting_model: str
        Path for the pre-existing model if retrain==True.

    =================== Results ===================================
    model: A trained model
    hist: the history of the training containing the losses, the validation losses etc.
    loss: 1D array of float, the loss through epoch.
    val_loss: 1D array of float, the validation loss through epoch.
    """
    if retrain:
        with kr.utils.CustomObjectScope({'OrderMap': nnhealpix.layers.OrderMap}):
            model = kr.models.load_model(preexisting_model)
    else:
        model = kr.models.Model(inputs=inputs, outputs=out)
        model.compile(loss=kr.losses.mse,
                      optimizer='adam',
                      metrics=[kr.metrics.mean_absolute_percentage_error])
    model.summary()

    # Set the callbacks
    # Save weights during training
    checkpointer_mse = kr.callbacks.ModelCheckpoint(
        filepath=out_dir + today + '_weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        period=1)

    # Stop the training if doesn't improve after 20 epochs
    stop = kr.callbacks.EarlyStopping(monitor='val_loss',
                                      verbose=0,
                                      restore_best_weights=True,
                                      patience=20)

    callbacks = [checkpointer_mse, stop]

    # Train the model
    hist = model.fit(x_train, y_train,
                     epochs=epoch,
                     batch_size=batch_size,
                     validation_data=(x_test, y_test),
                     verbose=1,
                     callbacks=callbacks,
                     shuffle=True)

    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    return model, hist, loss, val_loss


def plot_losses(loss, val_loss):
    """
    Show the loss and the validation loss of a model training
    ========= Parameters =======================
    hist: An array with the data of the training
    
    ========= Return ===========================
    Show the loss and the validation loss
    """
    print("min loss=", min(loss))
    print("min validation loss=", min(val_loss))

    plt.figure(1, figsize=(10, 10))
    plt.plot(loss, ':', color='blue', label="loss")
    plt.plot(val_loss, '--', color='green', label="val_loss")
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title("Losses and val_losses as function of epochs")
    plt.legend(loc=1)
    plt.yscale('log')
    plt.show()
    return


def make_prediction(inp, model, sigma_n=0):
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
    pred = model.predict(inp)
    return pred


def plot_error(pred, y_test):
    """
    Calculate and plot the histogram of the error

     ================= Parameters ==================
    pred: An array with the data predict by a model
    y_test: An array with the expected data
            should have the same shape as pred

    ================= Return ======================
    Show the plot of the error histogram
    """
    err = (pred[:, :] - y_test[:, :])  # /y_test[:,:]*100.
    mean_err = np.mean(abs(err))
    print('Mean error: ', mean_err)
    err = np.ravel(err)
    plt.figure(1, figsize=(10, 10))
    plt.hist(err, bins=100, alpha=0.5, label='err', color='blue')
    plt.title("Histogram of the error between the predicted and expected data")
    plt.xlabel("Difference between the predicted and expected data")
    plt.ylabel("Amount of points")
    plt.legend(loc=1)
    plt.show()
    return


def plot_chi2(pred, y_test):
    """
    Calculate and plot the histogram of the chi2.

    ================= Parameters ==================
    pred: An array with the data predict by a model
    y_test: An array with the expected data

    ================= Return ======================
    Show the plot of the chi2 histogram
    """
    chi2 = np.sum((pred - y_test) ** 2, axis=1)
    plt.figure(1, figsize=(10, 10))
    plt.hist(chi2, color="blue", bins=100, alpha=0.5, label='$\chi^2$')
    plt.title("$\chi^2$")
    plt.legend(loc=1)
    plt.show()
    return


def plot_in2out(pred, y_test):
    """
    ================= Parameters ==================
    pred: A 1D array or a list with the data predict by a model
    y_test: A 1D array or a list with the expected data

    ================= Return ======================
    Show the plot of the prediction as a function of the test data
    """
    plt.figure(1, figsize=(10, 10))
    plt.plot(y_test, pred, ' ', marker='.', label="20ep", color='blue')
    plt.plot(np.linspace(min(y_test), max(y_test), int(max(y_test) - min(y_test)) + 1),
             np.linspace(min(y_test), max(y_test), int(max(y_test) - min(y_test)) + 1), lw=2, label="Expected",
             color='green')
    plt.xlabel("Test data")
    plt.ylabel("Predicted data")
    plt.legend(loc=4)
    plt.show()
    return
