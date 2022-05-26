from cobaya.likelihood import Likelihood
from .binned_poisson_data import BinnedPoissonData

class BinnedPoissonLikelihood(Likelihood):

    def initialize(self):

        delNcat, delN2Dcat = self._get_data()
        self.data = BinnedPoissonData(delNcat, delN2Dcat)

    def get_requirements(self):
        if self.choose_theory == "camb":
            req = {"Hubble":  {"z": self.zz},
                   "H0": None, # H0 is derived
                   "Pk_interpolator": {"z": np.linspace(0, 3., 140), # should be less than 150 for camb
                                       "k_max": 4.0,
                                       "nonlinear": False,
                                       "hubble_units": False, # CLASS doesn't like this
                                       "k_hunit": False, # CLASS doesn't like this
                                       "vars_pairs": [["delta_nonu", "delta_nonu"]]}}
        elif self.choose_theory == "class":
            req = {"Hubble":  {"z": self.zz},
                   "Pk_interpolator": {"z": np.linspace(0, 3., 100), # should be less than 110 for class
                                       "k_max": 4.0,
                                       "nonlinear": False,
                                       "vars_pairs": [["delta_nonu", "delta_nonu"]]}}
        return req

    def _get_data(self):
        raise NotImplementedError

    def _get_theory(self, pk_intp, **kwargs):
        raise NotImplementedError

    def logp(self, **params_values):
        pk_intp = self.theory.get_Pk_interpolator(("delta_nonu", "delta_nonu"), nonlinear=False)
        theory = self._get_theory(pk_intp, **params_values)
        return self.data.loglike(theory)
