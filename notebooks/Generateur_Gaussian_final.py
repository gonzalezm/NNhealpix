from functions import CreateGaussianMapsAndCl
import numpy as np
import os
import sys
import datetime

name = sys.argv[1]

# Directory where files will be saved
dir = sys.argv[2]
today = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
out_dir = dir + '/{}/'.format(today)

try:
    os.makedirs(out_dir)
except:
    print("path error")
    pass

Nmodel=sys.argv[3]
sigma_p=5
Nside=16

l_p, C_l, Maps = CreateGaussianMapsAndCl(Nmodel, sigma_p, Nside)

np.save(out_dir + name + '_l_p', l_p)
np.save(out_dir + name + '_C_l', C_l)
np.save(out_dir + name + '_Maps', Maps)