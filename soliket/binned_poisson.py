from cobaya.likelihood import Likelihood
from .binned_poisson_data import BinnedPoissonData

class BinnedPoissonLikelihood(Likelihood):
    #name: str = "BinnedPoisson"

    def initialize(self):

        delNcat, delN2Dcat = self._get_data()
        self.data = BinnedPoissonData(delNcat, delN2Dcat)
        #print('\r :::::: this is initialisation in binned_poisson.py')

    def get_requirements(self):
        return {"Pk_interpolator": {"z": np.linspace(0, 2, 21),
                                    "k_max": 4.0,
                                    "nonlinear": False,
                                    "hubble_units": True,
                                    "k_hunit": True,
                                    "vars_pairs": [["delta_nonu", "delta_nonu"]]}}#,
                #"Hubble": {"z": np.linspace(0, 2, 21)}}

    def _get_data(self):
        #print("hello here is get_data in binned_poisson")
        raise NotImplementedError

    def _get_theory(self, pk_intp, **kwargs):
        #print("hello here is get_theory in binned_poisson")
        raise NotImplementedError

    def logp(self, **params_values):
        #print('\r :::::: computing logp in binned_poisson.py')
        pk_intp = self.theory.get_Pk_interpolator(("delta_nonu", "delta_nonu"), nonlinear=False)
        theory = self._get_theory(pk_intp, **params_values)
        print("logp params_values = ", params_values)
        return self.data.loglike(theory)
