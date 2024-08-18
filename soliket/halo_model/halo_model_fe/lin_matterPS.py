import numpy as np
import camb
from camb import model, initialpower

class matter_PS:
    def __init__(self, redshift, h, kmax, cosmo_param, cosmological_param):
        self.redshift           = redshift
        self.h                  = h
        self.kmax               = kmax
        self.cosmo_param        = cosmo_param
        self.cosmological_param = cosmological_param

    #computation of the lin matter PS
    def lin_matter_PS(self):
        print('Start the computation of the linear matter power spectrum')
        par_h = self.h * 100
        par_omega_b = self.cosmo_param.om_b * self.h ** 2
        par_omega_m = self.cosmo_param.om_m * self.h ** 2

        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=par_h, ombh2=par_omega_b, omch2=par_omega_m - par_omega_b, tau=self.cosmological_param['tau']
        )
        pars.InitPower.set_params(ns=self.cosmological_param['ns'], As=self.cosmological_param['As'], pivot_scalar=self.cosmological_param['pivot_scalar'])
        pars.set_matter_power(redshifts=self.redshift, kmax=self.kmax)

        pars.NonLinear = model.NonLinear_none
        results        = camb.get_results(pars)
        k_array, z, Pk_array = results.get_matter_power_spectrum(
            minkh=1e-3, maxkh=self.kmax, npoints=100
        )    
        return k_array, z, Pk_array