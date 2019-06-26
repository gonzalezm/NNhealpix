import math
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

directory = input("Enter directory where you want save your files (tape Enter to save in the same folder than this program): ")
l=np.linspace(5.0, 100.0, 2000)
l_p= (50.0-5.0)*np.random.random_sample(100000,) + 5
moy_l_p = np.mean(l_p)
ecart_l_p = np.sqrt(np.var(l_p))
sigma_p=5.0
print(l.shape, l_p.shape)
print(moy_l_p, ecart_l_p)
print(np.max(l_p))
np.save(directory + "l_p", l_p)

C_l=np.zeros((len(l),len(l_p)))
for i in range (len(l)):
    for j in range (len(l_p)):
        C_l[i,j]=np.exp(-((l[i]-l_p[j])**2.0)/(2.0*sigma_p**2.0))+10.0**(-5.0)
np.save(directory + "C_l", C_l)

nside = 16
Maps = []
for j in range (len(l_p)) :
    Map = hp.sphtfunc.synfast(C_l[:,j], nside)
    Maps = np.append(Maps,Map)
Maps = Maps.reshape((12*nside**2, len(l_p)))
np.save(directory + "Maps", Maps)

