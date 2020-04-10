import numpy as np
import solike.clusters.survey as Survey

from cobaya.likelihood import Likelihood

from .poisson_data import PoissonData


class PoissonLikelihood(Likelihood):
    name = "Cluster"
    data_path = None
    data_name = None
    columns = None

    def initialize(self):
        catalog = self._get_catalog()
        if self.columns is None:
            self.columns = catalog.columns
        self.data = PoissonData(self.name, catalog, self.columns)

    def get_requirements(self):
        return {'Pk_interpolator': {'z': np.linspace(0, 2, 41), 'k_max': 5.0,
                                    'nonlinear': False, 'hubble_units': True, 'k_hunit': True,
                                    'vars_pairs': [['delta_nonu', 'delta_nonu']]},
                'H': {'z': np.linspace(0, 2, 41)}}

    def _get_catalog(self):
        catalog = Survey.SurveyData(self.data_path, self.data_name)
        return catalog

    def _get_rate_fn(self, **kwargs):
        """Returns a callable rate function that takes each of 'columns' as kwargs.
        """
        raise NotImplementedError

    def _get_n_expected(self, **kwargs):
        """Computes and returns the integral of the rate function 
        """
        raise NotImplementedError

    def logp(self, **params_values):
        rate_fn = self._get_rate_fn(**params_values)
        n_expected = self._get_n_expected(**params_values)
        return self.data.loglike(rate_fn, n_expected)
