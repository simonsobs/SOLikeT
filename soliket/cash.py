import numpy as np
from typing import Optional

from cobaya.likelihood import Likelihood
from .cash_data import CashCData


class CashCLikelihood(Likelihood):
    name: str = "Cash-C"
    datapath = Optional[str]

    def initialize(self):

        x, N = self._get_data()
        self.data = CashCData(self.name, N)

    def _get_data(self):
        data = np.loadtxt(self.datapath, unpack=False)
        N = data[:, -1] # assume data stored like column_stack([z, q, N])
        x = data[:, :-1]
        return x, N

    def _get_theory(self, **kwargs):
        if("param_test_cash" in kwargs):
            return np.arange(kwargs["param_test_cash"])
        else:
            raise NotImplementedError

    def logp(self, **params_values):
        theory = self._get_theory(**params_values)
        return self.data.loglike(theory)
