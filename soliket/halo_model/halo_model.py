import numpy as np
from typing import Optional
from cobaya.theory import Theory
from cobaya.typing import InfoDict
import pyhalomodel as halo


class HaloModel(Theory):
    """Parent class for Halo Models."""
    
    def _get_Pk_mm_lin(self):
        for pair in self._var_pairs:
            self.k, self.z, Pk_mm = \
                self.provider.get_Pk_grid(var_pair=pair, nonlinear=False)

        return Pk_mm

    def get_Pk_mm_grid(self):

        return self._current_state("Pk_mm_grid")

    def get_Pk_gg_grid(self):

        return self._current_state("Pk_gg_grid")

    def get_Pk_gm_grid(self):

        return self._current_state("Pk_gm_grid")

class HaloModel_pyhm(HaloModel):
    """Halo Model wrapping the simple pyhalomodel code of Asgari, Mead & Heymans"""
    
    def initialize(self):

        self._var_pairs = set(("delta_tot", "delta_tot"))
        self.hmf_name = 'Tinker et al. (2010)'
        self.hmf_Dv = 330.
        self.Ms =
        self.z =
        self.k = 

    def get_requirements(self):

        return {}

    def must_provide(self):

        self.kmax = max(self.kmax, options.get("kmax", self.kmax))
        self.z = np.unique(np.concatenate(
                    (np.atleast_1d(options.get("z", self._default_z_sampling)),
                    np.atleast_1d(self.z))))

        needs = {}

        needs["Pk_grid"] = {
                "vars_pairs": self._var_pairs,
                "nonlinear": (False, False),
                "z": self.z,
                "k_max": self.kmax
            }

        needs["sigma_R"] = {"vars_pairs": self._var_pairs,
                           "z": self.z,
                           "k_max": self.kmax,
                           "R": self.R}


        return needs

    def calculate(self):

        Omega_m = self.provider.get_Omega_b(z) + self.provider.get_Omega_cdm(z) + self.provider.get_Omega_nu_massive(z)
        sigmaRs = self.provider.get_sigmaR(Rs, hubble_units=True, return_R_z=False)[[z].index(z)]

        hmod = halo.model(z, , name=self.hmf_name, Dv=self.hmf_Dv)

        rvs = hmod.virial_radius(Ms)
        cs = 7.85*(Ms/2e12)**-0.081*(1.+z)**-0.71
        Uk = win_NFW(ks, rvs, cs)
        matter_profile = halo.profile.Fourier(self.k, Ms, Uk, amplitude=Ms, normalisation=hmod.rhom, mass_tracer=True)

        Pk_mm_lin = self._get_Pk_mm_lin()

        Pk_2h, Pk_1h, Pk_hm = hmod.power_spectrum(self.k, Pk_mm_lin, self.Ms, sigmaRs, {'m': matter_profile}, verbose=True)

        state['Pk_mm_grid'] = Pk_hm['m-m']
        # state['Pk_gm_grid'] = Pk_hm['g-m']
        # state['Pk_gg_grid'] = Pk_hm['g-g']

    def win_NFW(k, rv, c):
        from scipy.special import sici
        rs = rv/c
        kv = np.outer(k, rv)
        ks = np.outer(k, rs)
        Sisv, Cisv = sici(ks+kv)
        Sis, Cis = sici(ks)
        f1 = np.cos(ks)*(Cisv-Cis)
        f2 = np.sin(ks)*(Sisv-Sis)
        f3 = np.sin(kv)/(ks+kv)
        f4 = np.log(1.+c)-c/(1.+c)
        Wk = (f1+f2-f3)/f4
        return Wk


    def Pk_1h_mm(self):

        return

    def Pk_2h_mm(self):

        Pk_mm_lin = self._get_Pk_mm_lin()

        return Pk_mm_lin * integral

    
    def Pk_gg(self):

        return self.Pk_cc_gg() + 2. * self.Pk_cs_gg() + self.Pk_ss_gg()

    def Pk_cc_gg(self):

        return integral

    def Pk_cs_gg(self):

        return integral

    def Pk_ss_gg(self):

        return integral



