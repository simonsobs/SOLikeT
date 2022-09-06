import pandas as pd

from cobaya.likelihood import Likelihood

from .poisson_data import PoissonData


class PoissonLikelihood(Likelihood):
    name = "Poisson"
    data_path = None
    columns = None

    def initialize(self):
        # print('initializing poisson')
        catalog = self._get_catalog()
        if self.columns is None:
            self.columns = catalog.columns
        self.data = PoissonData(self.name, catalog, self.columns)

    def get_requirements(self):
        return {}

    def _get_catalog(self):
        catalog = pd.read_csv(self.data_path)
        return catalog

    def _get_rate_fn(self, pk_intp,**kwargs):
        """Returns a callable rate function that takes each of 'columns' as kwargs.
        """
        raise NotImplementedError

    def _get_n_expected(self, **kwargs):
        """Computes and returns the integral of the rate function
        """
        raise NotImplementedError

    def logp(self, **params_values):
        pk_intp = self.theory.get_Pk_interpolator(("delta_nonu", "delta_nonu"), nonlinear=False)
        rate_fn = self._get_rate_fn(pk_intp,**params_values)
        n_expected = self._get_n_expected(pk_intp,**params_values)

        return self.data.loglike(rate_fn, n_expected)
