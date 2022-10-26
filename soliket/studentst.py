import numpy as np
from typing import Optional

from .gaussian import GaussianLikelihood
from .studentst_data import StudentstData


class StudentstLikelihood(GaussianLikelihood):
    name: str = "Students t"
    datapath = Optional[str]
    covpath: Optional[str] = None
    ncovsims: Optional[int] = None

    def logp(self, **params_values):
        theory = self._get_theory(**params_values)
        return self.data.loglike(theory)
