import numpy as np
import os
import sys
import datetime

import ConvNNTempLib as cnn

"""
parameters: name of the dataset, parent output directory, size of data to be generated
"""

name = sys.argv[1]

# Directory where files will be saved
out_dir = sys.argv[2]
today = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
#out_dir += '/{}-'.format(today) + name +  '/'
out_dir += '/' + name +  '/'

os.makedirs(out_dir, exist_ok=True)

nmodel = int(sys.argv[3])
lmax = 5
nside = 16

cl, maps = make_maps_with_random_spectra(nmodel, lmax, nside)

np.save(out_dir + 'cl', cl)
np.save(out_dir + 'maps', maps)
