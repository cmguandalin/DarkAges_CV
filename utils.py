import numpy as np
from scipy.interpolate import interp1d

# Loading HI coefficients from 1506.04152
z, T21_bar, Alpha, Beta, Gamma = np.loadtxt('coeffs_21cm.dat', unpack=True)

# Functions in mK
T21bar = interp1d(z,T21_bar)
alpha = interp1d(z,Alpha)
beta = interp1d(z,Beta)
gamma = interp1d(z,Gamma)



