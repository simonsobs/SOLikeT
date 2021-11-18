from cobaya.likelihood import Likelihood
from .binned_poisson_data import BinnedPoissonData

class BinnedPoissonLikelihood(Likelihood):

    def initialize(self):

        delNcat, delN2Dcat = self._get_data()
        self.data = BinnedPoissonData(delNcat, delN2Dcat)

    def get_requirements(self):
        return {"Pk_interpolator": {"z": np.linspace(0, 2, 21),
                                    "k_max": 4.0,
                                    "nonlinear": False,
                                    "hubble_units": True,
                                    "k_hunit": True,
                                    "vars_pairs": [["delta_nonu", "delta_nonu"]]},
                "Hubble": {"z": np.linspace(0, 2, 21)}}

    def _get_data(self):
        raise NotImplementedError

    def _get_theory(self, pk_intp, **kwargs):
        raise NotImplementedError

    def logp(self, **params_values):
        pk_intp = self.theory.get_Pk_interpolator(("delta_nonu", "delta_nonu"), nonlinear=False)
        theory = self._get_theory(pk_intp, **params_values)
        return self.data.loglike(theory)
