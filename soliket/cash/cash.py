from typing import Optional, Tuple
import numpy as np
from cobaya.likelihood import Likelihood
from .cash_data import CashCData


# Likelihood for independent Poisson-distributed data
# (here called Cash-C, see https://arxiv.org/abs/1912.05444)

class CashCLikelihood(Likelihood):
    name: str = "Cash-C"
    datapath: Optional[str] = None

    enforce_types: bool = True

    def initialize(self):
        x, N = self._get_data()
        self.data = CashCData(self.name, N)

    def _get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        data = np.loadtxt(self.datapath, unpack=False)
        N = data[:, -1]  # assume data stored like column_stack([z, q, N])
        x = data[:, :-1]
        return x, N

    def _get_theory(self, **kwargs) -> np.ndarray:
        if "cash_test_logp" in kwargs:
            return np.arange(kwargs["cash_test_logp"])
        else:
            raise NotImplementedError

    def logp(self, **params_values) -> float:
        theory = self._get_theory(**params_values)
        return self.data.loglike(theory)
