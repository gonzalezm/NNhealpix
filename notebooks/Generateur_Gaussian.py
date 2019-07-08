import sys
import os
import datetime
import numpy as np
import healpy as hp
import scipy.stats as stats


# This program create Gaussian spectra and CMB maps thanks this spctra
# It takes the name to give to data, the repository where save them 
# and the sigma of the gaussian noise
# It saves maps, spectra and the mean of the spectra

# Simulation name
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

# Number of map you want
nmap = np.int(sys.argv[3])

nside = 16

# Create random gaussian mean for the spectra
# between 5 and 50
l_p = 15 * np.random.random_sample(nmap,) + 5
#noise = np.random.randn(12*nside**2,nmap)*float(sys.argv[4])
#Take some informations about the l_p
moy_l_p = np.mean(l_p)
ecart_l_p = np.std(l_p)
max_lp = np.max(l_p)

print('l_p shape = {}'.format(l_p.shape))
print('l_p mean, std and max : {0} {1} {2}'.format(moy_l_p, ecart_l_p, max_lp))

# Create spectra (Cl) and associated maps
sigma_p = 5.0
l = np.linspace(5.0, 100.0, 2000)
print('l shape = {}'.format(l.shape))
C_l = np.empty((len(l), len(l_p)))
Maps = np.empty((12 * nside ** 2, len(l_p)))
for j, lp in enumerate(l_p):
    C_l[:, j] = stats.norm.pdf(l, lp, sigma_p) + 10.**(-5)
    Maps[:, j] = hp.sphtfunc.synfast(C_l[:, j], nside, verbose = 0)
Maps=Maps #+noise
# Save lp, Cl, maps in 3 files
np.save(out_dir + name + '_l_p', l_p)
np.save(out_dir + name + '_C_l', C_l)
np.save(out_dir + name + '_Maps', Maps)
