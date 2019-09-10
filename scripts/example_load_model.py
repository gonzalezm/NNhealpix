import keras as kr
import nnhealpix.layers as nn
import numpy as np

mymaps = np.load("/home/stagiaire/NNhealpix/notebooks" + "/mymaps.npy")

out_dir = "/home/stagiaire/NNhealpix/notebooks/20190705_10_34_56"

with kr.utils.CustomObjectScope({'OrderMap':nn.OrderMap}):
    load=kr.models.load_model(out_dir + "/mymodel.h5py.File")

mymaps = mymaps.reshape(mymaps.shape[0], len(mymaps[0]), 1)

prediction = load.predict(mymaps)