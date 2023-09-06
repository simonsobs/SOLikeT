from cobaya.likelihood import Likelihood
from .cash_data import CashCData


class CashCLikelihood(Likelihood):
    name: str = "Cash-C"

    def initialize(self):
        N = self._get_data()
        self.data = CashCData(self.name, N)

    def _get_data(self):
        raise NotImplementedError

    def _get_theory(self, **kwargs):
        if ("cash_test_logp" in kwargs):
            return np.arange(kwargs["cash_test_logp"])
        else:
            raise NotImplementedError

    def logp(self, **kwargs):
        pk_intp = self.theory.get_Pk_interpolator()
        theory = self._get_theory(pk_intp, **kwargs)
        return self.data.loglike(theory)
