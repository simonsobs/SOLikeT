from cobaya.likelihood import Likelihood
from .poisson_data import PoissonData


class PoissonLikelihood(Likelihood):
    name = "Poisson"
    data_path = None
    columns = None

    def initialize(self):
        self.data = PoissonData(self.name, self.catalog, self.columns)
        return {}

    def get_requirements(self):
        return {}

    def _get_rate_fn(self, **kwargs):
        """Returns a callable rate function that takes each of 'columns' as kwargs.
        """
        raise NotImplementedError

    def _get_n_expected(self, **kwargs):
        """Computes and returns the integral of the rate function
        """
        raise NotImplementedError

    def logp(self, **kwargs):

        pk_intp = self.theory.get_Pk_interpolator()
        rate_densities = self._get_rate_fn(pk_intp, **kwargs)
        n_expected = self._get_n_expected(pk_intp, **kwargs)

        return self.data.loglike(rate_densities, n_expected)
