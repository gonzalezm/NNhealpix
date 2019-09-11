import keras as kr
import healpy as hp
import nnhealpix
import nnhealpix.layers
import numpy as np
import math
import camb


def make_maps_with_gaussian_spectra(nmodel, sigma_p, nside):
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

    ll = np.arange(4 * nside)

    maps = np.empty((nmodel, hp.nside2npix(nside)))
    cl = np.empty((nmodel, len(ll)))
    lp = np.empty(nmodel)

    for i in range(nmodel):
        np.random.seed()
        lp[i] = np.random.uniform(5., 2. * nside)
        cl[i, :] = gaussian(ll, lp[i], sigma_p)
        maps[i, :] = hp.synfast(cl[i, :], nside, verbose=0)

    return lp, cl, maps


def make_maps_with_real_spectra(nmodels, nside):
    """
    Create a set of data with realistic maps of CMB, spectra and alm

    ============= Parameters ===================================
    nmodels: int
        Number of models.
    nside: int
        Resolution for the maps, power of 2.

    ================ Return ====================================
    mymaps: 2D array
        Set of maps of the full sky (realistic CMB maps), shape (nmodels, #pixels)
    mycls: 2D array
        Set of realistic spectra, shape (nmodels, #cls)
    expcls: 2D array
        Cls computed with anafast, shape (nmodels, #cls)
    myalms: complex array
        alms computed with anafast, shape (nmodels, #alms)
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


def AddWhiteNoise(maps, sigma_n):
    """
    Add a gaussian white noise on the map
    ============ Parameters =================
    maps: 2D array
        Set of maps of the full sky, shape(#maps, #pixels)
    sigma_n: A float, the standard deviation of the gaussian noise
    """
    return maps + np.random.randn(maps.shape[0], maps.shape[1]) * sigma_n


def NormalizeMaps(map):
    """
    Normalize a map
    """
    return (map - np.mean(map)) / np.std(map)


def make_patch_maps(maps, theta, phi, radius):
    """
    Transform a set of maps of the full sky in a set of map of a sky patch

    ============ Parameters ==============================
    maps: 2D array
        Set of maps of the full sky, shape(#maps, #pixels)
    theta: float,
        Angle in radian for the center of the patch
    phi: float
        Angle in radian for the center of the patch
    radius: float
        Radius of the patch in radian

    ============ Return ==================================
    map_patch: 2D array
        Maps with only one patch not unseen, shape(#maps, #pixels)
    """
    vec = hp.ang2vec(theta, phi)

    nside = hp.npix2nside(maps.shape[1])
    # make the map with only the patch from full maps
    patch = hp.query_disc(nside, vec, radius)
    map_patch = np.full((maps.shape[0], maps.shape[1]), hp.UNSEEN)
    for i in range(maps.shape[0]):
        map_patch[i, patch] = maps[i, patch]

    return map_patch


def make_model(nside, num_out, retrain=False, preexisting_model=None):
    """
    Architecture of the Neural Network using the NNhealpix functions

    ========= Parameters =============
    shape: tuple
        Shape of ONE input map.
    nside: int
        Resolution parameter of your input maps, must be a power of 2.
    num_out: int
        Number of neuron of the output layer.
    retrain: bool
        Set True if you want to load a model already trained.
    preexisting_model: str
        Path for the pre-existing model if retrain==True.

    ========= Return ================
    model
    """

    if retrain:
        with kr.utils.CustomObjectScope({'OrderMap': nnhealpix.layers.OrderMap}):
            model = kr.models.load_model(preexisting_model)
    else:
        inputs = kr.layers.Input((12 * nside ** 2, 1))
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

        model = kr.models.Model(inputs=inputs, outputs=out)
        model.compile(loss=kr.losses.mse,
                      optimizer='adam',
                      metrics=[kr.metrics.mean_absolute_percentage_error])

    model.summary()

    return model


def make_training(model, x_train, y_train, validation_split, epochs, batch_size, out_dir, patience=10, today=None):
    """
    Train a model.

    =================== Parameters ================================
    x_train: 3D array of float
        Training input data.
    y_train: A 1 or 2D array or list of float
        Training output data.
    validation_split: float
        Fraction of the data used for validation, between 0 and 1.
    epochs: int
        Number of epochs.
    batch_size: int
        Batch size.
    out_dir: str
        Repository where the model and the weights will be saved.
    patience : int
        Number of epochs with no improvement after which training will be stopped.
    today: str
        Name for the weights that will be saved.


    =================== Results ===================================
    model: A trained model
    hist: the history of the training containing the losses, the validation losses etc.
    """
    x_train = np.expand_dims(x_train, axis=2)

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

    # Stop the training if doesn't improve
    stop = kr.callbacks.EarlyStopping(monitor='val_loss',
                                      verbose=0,
                                      restore_best_weights=True,
                                      patience=patience)

    callbacks = [checkpointer_mse, stop]

    # Train the model
    hist = model.fit(x_train, y_train,
                     epochs=epochs,
                     batch_size=batch_size,
                     validation_split=validation_split,
                     verbose=1,
                     callbacks=callbacks,
                     shuffle=True)

    # Save the model as a pickle in a file
    kr.models.save_model(model, out_dir + today + '_model.h5py.File')

    return model, hist


def make_prediction(inp, model, sigma_n=0):
    """
    Make a prediction with given model and data
    Add also noise to the data.

    ============ Parameters =============================
    inp: array
        Input data.
    model: The given model
    sigma_n: float
        Standard deviation of the gaussian white noise.

    ============ Return =================================
    pred: An array or a list with the predicted data.
    """
    inp = AddWhiteNoise(inp, sigma_n)
    inp = NormalizeMaps(inp)
    inp = inp.reshape(inp.shape[0], inp.shape[1], 1)
    pred = model.predict(inp)
    return pred


# def ConvNNhealpix(shape, nside, num_out):
#     """
#     Architecture of the Neural Network using the NNhealpix functions
#
#     ========= Parameters =============
#     shape: tuple
#         Shape of ONE input map.
#     nside: int
#         Resolution parameter of your input maps, must be a power of 2.
#     num_out: int
#         Number of neuron of the output layer.
#
#     ========= Return ================
#     inputs: keras tensor
#         Input used to build the network.
#     out: keras tensor
#         Last layer of the network.
#     """
#     inputs = kr.layers.Input(shape)
#     x = inputs
#
#     # NBB loop (conv and degrade from nside to 1)
#
#     for i in range(int(math.log(nside, 2))):
#         # Recog of the neighbours & Convolution
#         print(int(nside / (2 ** i)), int(nside / (2 ** (i + 1))))
#         x = nnhealpix.layers.ConvNeighbours(int(nside / (2 ** i)),
#                                             filters=32,
#                                             kernel_size=9)(x)
#         x = kr.layers.Activation('relu')(x)
#         # Degrade
#         x = nnhealpix.layers.AveragePooling(int(nside / (2 ** i)),
#                                             int(nside / (2 ** (i + 1))))(x)
#
#     # End of the NBBs
#
#     x = kr.layers.Dropout(0.2)(x)
#     x = kr.layers.Flatten()(x)
#     x = kr.layers.Dense(256)(x)
#     x = kr.layers.Activation('relu')(x)
#     x = kr.layers.Dense(num_out)(x)
#
#     out = kr.layers.Activation('relu')(x)
#
#     return inputs, out


# def PreprocessML(inp, outp, ntest, ntrain, patch=False, theta=0, phi=0, r=0):
#     """
#     Prepare the data for the ML with ND inputs and outputs
#
#     ========= Parameters ===============
#     inp:2D array
#         Set of maps of the full sky, first index is the number of the map
#         second index is the number of the pixel on the map.
#         The input of the machine learning
#     outp: 1 or 2-D array
#           inp.shape[0] must be equal to outp.shape[0]
#     ntest: float
#         Fraction for validation data between 0 and 1.
#     sigma_n: float
#         She standard deviation of the gaussian white noise.
#     patch: bool
#         Say if you want a patch of the sky (True) or the full sky (False)
#     theta: float,
#         Angle in radian for the center of the patch
#     phi: float
#         Angle in radian for the center of the patch
#     radius: float
#         Radius of the patch in radian
#
#     ========= Return ==================
#     x_train: A 3D array of float, the training input data
#     y_train: A 1 or 2D array or list of float, the training output data
#     x_test: A 3D array of float, the validation input data
#     y_test: A 1 or 2D array of float, Validation output data
#     num_out: int
#         Number of neurons of the output layer.
#     shape: tuple
#         Shape of ONE input map.
#     """
#
#     nmodel = outp.shape[0]
#     if ntest != int(ntest) and 0.0 <= ntest <= 1.0:
#         ntest = int(ntest * nmodel)
#         print("ntest={}".format(ntest))
#     if ntrain != int(ntrain) and 0.0 <= ntrain <= 1.0:
#         ntrain = int(ntrain * nmodel)
#         print("ntrain={}".format(ntrain))
#
#     if len(outp.shape) == 1:
#         outp = outp.reshape(outp.shape[0], 1)
#     num_out = outp.shape[1]
#
#     if patch:
#         inp = MakePatchMaps(inp, theta, phi, r)
#
#     # Split between train and test
#     x_train = inp[:ntrain, :]
#     y_train = outp[0:ntrain, :]
#     x_test = inp[ntrain:(ntrain + ntest), :]
#     y_test = outp[ntrain:(ntrain + ntest), :]
#
#     x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
#     x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
#
#     shape = (inp.shape[1], 1)
#
#     return x_train, y_train, x_test, y_test, num_out, shape


# def MakeAndTrainModel(x_train, y_train,
#                       x_test, y_test,
#                       epoch, batch_size,
#                       out_dir, today=None,
#                       inputs=None, out=None,
#                       retrain=False, preexisting_model=None):
#     """
#     Create a model from a Network, show this network and train the model.
#
#     =================== Parameters ================================
#     x_train: 3D array of float
#         Training input data.
#     y_train: A 1 or 2D array or list of float
#         Training output data.
#     x_test: A 3D array of float
#         Validation input data.
#     y_test: A 1 or 2D array of float
#         Validation output data.
#     epoch: int
#         Number of epochs.
#     batch_size: int
#         Batch size.
#     out_dir: str
#         Repository of the saved weights.
#     today: str
#         Name for the weights that will be saved.
#     inputs: keras tensor
#         Input used to build the network if retrain=False.
#     out: keras tensor
#         Last layer of the network if retrain=False.
#
#
#     =================== Results ===================================
#     model: A trained model
#     hist: the history of the training containing the losses, the validation losses etc.
#     loss: 1D array of float, the loss through epoch.
#     val_loss: 1D array of float, the validation loss through epoch.
#     """
#     if retrain:
#         with kr.utils.CustomObjectScope({'OrderMap': nnhealpix.layers.OrderMap}):
#             model = kr.models.load_model(preexisting_model)
#     else:
#         model = kr.models.Model(inputs=inputs, outputs=out)
#         model.compile(loss=kr.losses.mse,
#                       optimizer='adam',
#                       metrics=[kr.metrics.mean_absolute_percentage_error])
#     model.summary()
#
#     # Set the callbacks
#     # Save weights during training
#     checkpointer_mse = kr.callbacks.ModelCheckpoint(
#         filepath=out_dir + today + '_weights.{epoch:02d}-{val_loss:.2f}.hdf5',
#         monitor='val_loss',
#         verbose=1,
#         save_best_only=True,
#         save_weights_only=False,
#         mode='min',
#         period=1)
#
#     # Stop the training if doesn't improve after 20 epochs
#     stop = kr.callbacks.EarlyStopping(monitor='val_loss',
#                                       verbose=0,
#                                       restore_best_weights=True,
#                                       patience=20)
#
#     callbacks = [checkpointer_mse, stop]
#
#     # Train the model
#     hist = model.fit(x_train, y_train,
#                      epochs=epoch,
#                      batch_size=batch_size,
#                      validation_data=(x_test, y_test),
#                      verbose=1,
#                      callbacks=callbacks,
#                      shuffle=True)
#
#     loss = hist.history['loss']
#     val_loss = hist.history['val_loss']
#
#     return model, hist, loss, val_loss
