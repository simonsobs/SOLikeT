from astropy.cosmology import Planck15, Planck18
import numpy as np

cosmo     = Planck18

#set physical constants
c  = 299792458.e-3

class cosmo_param:
    def __init__(self, redshift, cosmological_param, cosmo, default_lin_matter_PS):
        self.redshift              = redshift
        self.cosmological_param    = cosmological_param
        self.cosmo                 = cosmo
        self.default_lin_matter_PS = default_lin_matter_PS
        
        self.compute_params()

    def compute_params(self):
        cosmo = self.cosmo
        h     = cosmo.h
        H_0   = cosmo.H(0).value
        om_l  = cosmo.Ode0
        om_m  = cosmo.Om0
        om_b  = cosmo.Ob0

        par_h       = h * 100
        par_omega_m = om_m * h ** 2
        par_omega_b = om_b * h ** 2

        D_a   = cosmo.angular_diameter_distance(self.redshift).value
        D_h   = c / H_0
        dV_dz = D_h * ((1 + self.redshift) ** 2 * D_a ** 2) / np.sqrt(om_l + (1 + self.redshift) ** 3 * om_m) / h ** -3

        self.om_b = om_b
        self.om_m = om_m
        
        return h, dV_dz

    def read_matter_PS(self):
        if self.cosmological_param['matter_PS'] is None:
            print('Using default linear matter power spectrum with CAMB')
            matter_PS = np.loadtxt(self.default_lin_matter_PS)
            k_array_T = matter_PS[:,0]
            k_array   = np.transpose(k_array_T)

            Pk_array_T = matter_PS[:,1:]
            Pk_array   = np.transpose(Pk_array_T)
        else:
            print('Using external linear matter power spectrum provided by the user')
            matter_PS = np.loadtxt(self.cosmological_param['matter_PS'])
            k_array_T = matter_PS[:,0]
            k_array   = np.transpose(k_array_T)

            Pk_array_T = matter_PS[:,1:]
            Pk_array   = np.transpose(Pk_array_T)

        return k_array, Pk_array