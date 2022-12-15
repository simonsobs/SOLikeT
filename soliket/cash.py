import numpy as np
from typing import Optional

from cobaya.likelihood import Likelihood
from .cash_data import CashCData


class CashCLikelihood(Likelihood):
    name: str = "Cash-C"
    datapath = Optional[str]

    def initialize(self):
        N = self._get_data()
        self.data = CashCData(self.name, N)

    def _get_data(self):
        raise NotImplementedError

    def _get_theory(self, pk_intp, **kwargs):
        raise NotImplementedError

    def logp(self, **params_values):
        pk_intp = self.theory.get_Pk_interpolator(("delta_nonu", "delta_nonu"), nonlinear=False)
        theory = self._get_theory(pk_intp, **params_values)
        return self.data.loglike(theory)
