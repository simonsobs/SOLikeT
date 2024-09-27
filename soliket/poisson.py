from cobaya.likelihood import Likelihood
from .poisson_data import PoissonData


class PoissonLikelihood(Likelihood):
    name: str = "Poisson"

    def initialize(self):
        catalog, columns = self._get_catalog()
        self.data = PoissonData(self.name, catalog, columns)

    def _get_catalog(self):
        raise NotImplementedError

    def _get_rate_fn(self, **kwargs):
        """Returns a callable rate function that takes each of 'columns' as kwargs.
        """
        raise NotImplementedError

    def _get_n_expected(self, **kwargs):
        """Computes and returns the integral of the rate function
        """
        raise NotImplementedError

    def logp(self, **kwargs):
        pk_intp = self.provider.get_Pk_interpolator()
        rate_densities = self._get_rate_fn(pk_intp, **kwargs)
        n_expected = self._get_n_expected(pk_intp, **kwargs)
        return self.data.loglike(rate_densities, n_expected)
