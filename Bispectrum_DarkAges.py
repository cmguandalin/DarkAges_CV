"""
    Author: Caroline Guandalin
    Based on the code of Guandalin et al.: 2112.05034

"""

import numpy as np
from scipy.integrate import simps, quad
from scipy.interpolate import interp1d
from scipy.special import erf, gamma
from classy import Class
import utils as ut
import math
import sys
from tqdm.auto import tqdm

print('[Initialising CLASS]\n')

c = 299792.458 #km/s
h = 0.674
ns = 0.965
s8 = 0.811
As = np.exp(3.0448)/1e10
Om = 0.315
Ob = 0.049
Ok = 0.000
N_ur = 2.99
N_ncdm = 0.0
w = -1.0

print('(Pre-defined cosmology)')
print('      h:', h)
print('    n_s:', ns)
print('     s8:', s8)
print('   N_ur:', N_ur)
print(' N_ncdm:', N_ncdm)
print('Omega_m:', Om)
print('      w:', w )
print('\n')

# Initialise CLASS (linear power spectra)
class_settings = {'output': 'mPk,mTk', 
                  'lensing': 'no',
                  'h': h, 
                  'n_s': ns,
                  #'sigma8': s8, 
                  'A_s': As,
                  'Omega_cdm': Om-Ob, 
                  'Omega_b': Ob,
                  'z_max_pk': 200.0,
                  'P_k_max_1/Mpc': 250}
pclass = Class()
pclass.set(class_settings)
pclass.compute()

# Get background quantities
bg       = pclass.get_background()
H        = interp1d(bg['z'],(c/h)*bg['H [1/Mpc]'])
comov    = interp1d(bg['z'],h*bg['comov. dist.'])
redshift = interp1d(h*bg['comov. dist.'],bg['z'])
Dz       = interp1d(bg['z'],bg['gr.fac. D'])

print('[CLASS Initialised]')

S1 = {
    "name" : "Stage 1",
    "t_total" : 8e7/60/60, # hours
    "im_type" : "interferometer",
    "omega_sky" : 20000.0,
    "fsky" : 20000.0/41251.83
}

S2 = {
    "name" : "Stage 2",
    "t_total" : 8e7/60/60, # hours
    "im_type" : "interferometer",
    "omega_sky" : 20000.0,
    "fsky" : 20000.0/41251.83
}

S3 = {
    "name" : "Stage 3",
    "t_total" : 8e7/60/60, # hours
    "im_type" : "interferometer",
    "omega_sky" : 20000.0,
    "fsky" : 20000.0/41251.83
}

# Bispectrum Dark Ages class

class fisher(object):
    def __init__(self, redshift, freq_res=0.01, t_survey=8e7, wedge='hor', window='none', IM='S1'):
        
        # Frequency resolution in MHz
        self.freq_res = freq_res
        # Centre of the redshift bin
        self.zcen = redshift
        
        # Cosmological quantities
        self.chi  = comov(self.zcen)
        self.HMpc = H(self.zcen)/c
        self.kvec = np.logspace(-5.0, np.log10(250), 1000)
        self.Plin = interp1d(self.kvec/h,np.array([pclass.pk(_k, self.zcen) for _k in self.kvec])*(h**3.0),kind='cubic')
        # Get baryon transfer function for P_HI
        ktmp    = pclass.get_transfer(self.zcen)['k (h/Mpc)']
        self.Tb = interp1d(ktmp,pclass.get_transfer(self.zcen)['d_b'])
        
        # Non-linear scale k_max (astro-ph/0207664)
        #self.k_max = 0.14*(1+self.zcen)**(2.0/(2+ns))
        
        # HI specifications
        self.IM = IM
        if self.IM == 'S1':
            HI = S1
        elif self.IM == 'S2':
            HI = S2
        else:
            HI = S3

        self.im_type = HI["im_type"]
        Omega_sky    = HI["omega_sky"] # deg2
        Omega_tot    = Omega_sky * (np.pi/180)**2 # rad
        fsky         = HI["fsky"]
        
        nubar = 1420.4/(1.0+self.zcen) # MHz
        numax = nubar*1.15
        numin = nubar*0.85
        dnu  = numax-numin
        zmin = 1420.4/numax - 1.0
        zmax = 1420.4/numin - 1.0
        self.Volume = fsky*4*np.pi*(comov(zmax)**3 - comov(zmin)**3)/3.0 # (Mpc/h)^3
        self.k_min  = 2*np.pi/self.Volume**(1.0/3.0)
        
        r_nu = (1.0 + self.zcen)**2.0 / self.HMpc
        # Factor of 2 missing because the field is real!
        self.kpar_max = np.pi / (r_nu * (self.freq_res / 1420.4))
        self.kpar_min = np.pi / (r_nu * (dnu/1420.4))
        self.kFG = self.kpar_min
        #print('z=%.1f, kpar_min=%.5f, kpar_max=%.2f' %(self.zcen,self.kpar_min,self.kpar_max))
        
        # HI quantities for the Dark Ages (no bias) in K
        self.T21 = ut.T21bar(self.zcen)
        self.alpha = ut.alpha(self.zcen)
        self.beta = ut.beta(self.zcen)
        self.gamma = ut.gamma(self.zcen)

        ## Beam - used for single-dish experiments
        #self.FWHM = 0.21*(1+self.zcen)/D_dish
        #self.sigma_beam = self.FWHM/2.355
        #self.sig_perp = self.sigma_beam*self.chi

        # Foregrounds
        #self.kFG = kFG
        self.window = window
        self.wedge = wedge

        #############################
        # NOISE CONTRIBUTION
        #
        
        # INCLUDE PROPER CODE HERE  #
                                    #
        # END OF NOISE CONTRIBUTION #
        #############################
        
    # END of _init_

    def Pk_phi(self, k, k0=0.05, units=True):
        """
            Power spectrum of the Bardeen potential Phi in the matter-dominated era.
            k in units of h/Mpc.
        """
        k_pivot = k0/h
        resp = (9.0/25.0) * As * (k/k_pivot)**(ns - 1.0)
        if units:
            resp *= 2*np.pi**2.0/k**3.0
        return resp
    
    def M(self, k, k0=0.05):
        """
            The scaling factor between the primordial scalar power spectrum and 
            the late-time matter power spectrum
        """
        # Some bug in Python was not allowing to call Pk_phi(k) directly, so here's this aweful thing
        k_pivot = k0/h
        self.tmp_pkprim = ((9.0/25.0) * As * (k/k_pivot)**(ns - 1.0)) * (2*np.pi**2.0/k**3.0)
        self.tmp_pklin = self.Plin(k)
        tmp_mk = self.tmp_pklin/self.tmp_pkprim
        self.resp = np.sqrt(tmp_mk)
        
        return self.resp
        
    def P_b(self, k):
        return self.Tb(k)**2.0 * self.Pk_phi(k)*(25/9)
    
    def k_wedge(self, k_perp):
            
        if self.wedge == 'hor': 
            self.resp = k_perp*self.chi*self.HMpc/(1+self.zcen)
        elif self.wedge == 'pb':
            # Need the dish size to set FWHM
            self.resp = k_perp*np.sin(self.FWHM/2.0)*self.chi*self.HMpc/(1+self.zcen)
        else:
            self.resp = np.zeros_like(k_perp)
            
        return self.resp   
        
    def D_FG(self, k_para, k_perp, ind=1.0):
        alpha_FG  = 1.0
        wedge_FG  = np.heaviside(np.fabs(k_para)-self.k_wedge(k_perp),0.0)
        
        if self.window == 'HS_chi' or self.window == 'chi':
            window_FG = np.heaviside(np.fabs(k_para)-2*np.pi/self.chi,0.0)
            resp = alpha_FG * window_FG
        elif self.window == 'HS' or self.window == 'HS_kfg':
            window_FG = np.heaviside(np.fabs(k_para)-self.kFG,0.0)
            resp = alpha_FG * window_FG
        else:
            #resp = alpha_FG * ( 1 - np.exp(-(np.fabs(k_para)/self.kFG)**ind) )
            resp = 1.0

        return wedge_FG*resp
        
    def P_HI(self, k_para,k_perp):
        k = np.sqrt(k_para**2.0+k_perp**2.0)
        Pb = self.P_b(k)
        
        if wedge == 'hor' or wedge == 'pb':
            Dfg = self.D_FG(k_para,k_perp)
        else:
            Dfg = 1.0
        
        if self.im_type == 'single_dish':
            beam = self.D_beam(k_perp)
            self.resp = (Dfg*beam)**2.0 * (self.alpha + self.T21*(k_para/k)**2.0)**2.0 * Pb #+ SINGLE_DISH_NOISE
        else:
            beam = 1.0
            self.resp = (Dfg*beam)**2.0 * (self.alpha + self.T21*(k_para/k)**2.0)**2.0 * Pb #+ NOISE(P_kperp)

        return self.resp
    
    ###########################################
    # MAIN FUNCTION
    #
    def get_sigma_fNL(self, set_kmin = None, set_kmax = None,
                 nkperp_hi=32, nkperp_lo=16,
                 nkpara_hi=32, nkpara_lo=32,
                      verbose=False):
        
        # to set specific k_min and k_max, give a list:
        # set_kmin = [kpara_min,kperp_min] and set_kmax = [kpara_max,kperp_max]
        
        def int_FisherB(k1_para, k2_para, k1_perp, k2_perp, k3_perp):

            if (k3_perp <= np.fabs(k1_perp-k2_perp)) or (k3_perp >= k1_perp+k2_perp):
                return np.zeros_like(k1_para)
            
            k3_para = -(k1_para + k2_para)
            k1 = np.sqrt(k1_para**2.0+k1_perp**2.0)
            k2 = np.sqrt(k2_para**2.0+k2_perp**2.0)
            k3 = np.sqrt(k3_para**2.0+k3_perp**2.0)
            
            # Damping from foregrounds and wedge
            # Beam smoothing already present in the noise Nk_hh
            beams_HI = self.D_FG(k1_para,k1_perp)*self.D_FG(k2_para,k2_perp)*self.D_FG(k3_para,k3_perp)
            
            # Bispectrum part
            M1 = self.M(k1)
            M2 = self.M(k2)
            M3 = self.M(k3)

            P12 = self.Pk_phi(k1)*self.Pk_phi(k2)
            P13 = self.Pk_phi(k1)*self.Pk_phi(k3)
            P23 = self.Pk_phi(k2)*self.Pk_phi(k3)

            dBPhi_dfNL = 2.0*(P12 + P13 + P23)

            term1 = (self.alpha+self.T21*(k1_para/k1)**2.0)*M1
            term2 = (self.alpha+self.T21*(k2_para/k2)**2.0)*M2
            term3 = (self.alpha+self.T21*(k3_para/k3)**2.0)*M3
            
            dB_dfNL2 = (term1*term2*term3*dBPhi_dfNL)**2.0
            
            # Covariance part 
            Phh_1 = self.P_HI(k1_para,k1_perp)
            Phh_2 = self.P_HI(k2_para,k2_perp)
            Phh_3 = self.P_HI(k3_para,k3_perp)

            return beams_HI*(k1*k2*k3) * dB_dfNL2/(Phh_1*Phh_2*Phh_3)
        
        if set_kmin is None:
            kpara_min = self.k_min
            kperp_min = self.k_min
        else:
            kpara_min = set_kmin[0]
            kperp_min = set_kmin[1]
        if set_kmax is None:
            kpara_max = self.kpar_max
            kperp_max = 4.0
        else:
            kpara_max = set_kmax[0]
            kperp_max = set_kmax[1]
        
        # k_min of the survey for k_perp
        k1_perps = np.geomspace(kperp_min, kperp_max, 64)
        k2_perps = np.geomspace(kperp_min, kperp_max, 64)
        k3_perps = np.geomspace(kperp_min, kperp_max, 64)
        k1_pars  = np.geomspace(kpara_min, kpara_max, 64)
        k2_pars  = np.geomspace(kpara_min, kpara_max, 64)

        integ_k2par = np.zeros(len(k2_pars))
        integ_perp = np.zeros([len(k1_perps), len(k2_perps), len(k3_perps)])

        if verbose:
            for i1, k1 in tqdm(enumerate(k1_perps), total=len(k1_perps)):
                for i2, k2 in enumerate(k2_perps):
                    for i3, k3 in enumerate(k3_perps):
                        for j2, k2p in enumerate(k2_pars):
                            integ = int_FisherB(k1_pars, k2p, k1, k2, k3)
                            integ_k2par[j2] = simps(integ, x=k1_pars)
                        integ_perp[i1, i2, i3] = simps(integ_k2par, x=k2_pars)
        else:
            for i1, k1 in enumerate(k1_perps):
                for i2, k2 in enumerate(k2_perps):
                    for i3, k3 in enumerate(k3_perps):
                        for j2, k2p in enumerate(k2_pars):
                            integ = int_FisherB(k1_pars, k2p, k1, k2, k3)
                            integ_k2par[j2] = simps(integ, x=k1_pars)
                        integ_perp[i1, i2, i3] = simps(integ_k2par, x=k2_pars)

        # integrating over ks 
        integ_12 = simps(integ_perp, x=k3_perps, axis=-1)
        integ_1 = simps(integ_12, x=k2_perps, axis=-1)
        Fisher_fNL  = self.Volume*simps(integ_1, x=k1_perps)/(6*(2*np.pi)**2.0 * (2.0*np.pi)**3.0)
        sig_fNL = 1/np.sqrt(Fisher_fNL)
        
        return sig_fNL