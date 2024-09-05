import numpy as np
from cobaya.theory import Theory
from .utils import u_p_nfw_hmf_bias
from .HODS import hod_ngal
from .power_spectrum import mm_gg_mg_spectra
from .lin_matterPS import * 
from .cosmology import *


class HaloModel(Theory):
    _logz = np.linspace(-3, np.log10(1100), 150)
    _default_z_sampling = 10**_logz
    _default_z_sampling[0] = 0

    def initialize(self):
        self._var_pairs = set()
        self._required_results = {}
    
    def _get_Pk_mm_lin(self):
        for pair in self._var_pairs:
            self.k, self.z, Pk_mm = \
                self.provider.get_Pk_grid(var_pair=pair, nonlinear=False)

        return Pk_mm

    def get_Pk_mm_grid(self):

        return self.current_state["Pk_mm_grid"]

    def get_Pk_gg_grid(self):

        return self.current_state["Pk_gg_grid"]

    def get_Pk_gm_grid(self):

        return self.current_state["Pk_gm_grid"]
    

class HaloModel_fe(HaloModel):
    def initialize(self):
        super().initialize()
        self.logmass = np.linspace(self.Mmin, self.Mmax, self.nm)
        self.clust_param = {'sigma_EP': self.sigma_EP,
                            'sigma_LP': self.sigma_LP,
                            'scale_EP': self.scale_EP,
                            'scale_LP': self.scale_LP,
                            'alpha_EP': self.alpha_EP,
                            'alpha_LP': self.alpha_LP,
                            'LogMmin_EP': self.LogMmin_EP,
                            'LogMmin_LP': self.LogMmin_LP,
                            'Dc': self.Dc,
                            }

    def must_provide(self, **requirements):

        options = requirements.get("halo_model") or {}
        self._var_pairs.update(
            set((x, y) for x, y in
                options.get("vars_pairs", [("delta_tot", "delta_tot")])))

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
                           "R": np.linspace(0.14, 66, 256) # list of radii required
                           }
        return needs

    def calculate(self, state: dict, want_derived: bool = True,
                  **params_values_dict):
        Pk_mm_lin = self._get_Pk_mm_lin()

        self.instance_200 = u_p_nfw_hmf_bias(self.k, Pk_mm_lin, 
                                             self.logmass, self.z, self.Dc)
        self.instance_HOD = hod_ngal(self.logmass, self.z, 
                                     self.clust_param, self.instance_200)

        spectra = mm_gg_mg_spectra(
                    self.k,
                    Pk_mm_lin,
                    self.logmass,
                    self.z,
                    self.instance_HOD,
                    self.instance_200,
                    self.gal_mod
                )
        
        Pgal = spectra.halo_terms_galaxy()[0]

        state['Pk_gg_grid'] = Pgal