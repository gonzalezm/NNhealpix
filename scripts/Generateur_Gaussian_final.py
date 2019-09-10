import numpy as np
import os
import sys
import datetime

import ConvNNTempLib as cnn

name = sys.argv[1]

# Directory where files will be saved
dir = sys.argv[2]
today = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
out_dir = dir + '/{}/'.format(today)

Nmodel = int(sys.argv[3])
sigma_p = 5
Nside = 16

l_p, C_l, Maps = cnn.CreateGaussianMapsAndCl(Nmodel, sigma_p, Nside)

try:
    os.makedirs(out_dir)
except:
    print("path error")
    pass

np.save(out_dir + name + '_l_p', l_p)
np.save(out_dir + name + '_C_l', C_l)
np.save(out_dir + name + '_Maps', Maps)
