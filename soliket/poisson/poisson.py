from typing import Callable, List, Optional
import pandas as pd
from cobaya.likelihood import Likelihood

from .poisson_data import PoissonData


class PoissonLikelihood(Likelihood):
    name: str = "Poisson"
    data_path: Optional[str] = None
    columns: Optional[List[str]] = None

    def initialize(self) -> None:
        catalog = self._get_catalog()
        if self.columns is None:
            self.columns = catalog.columns
        self.data = PoissonData(self.name, catalog, self.columns)

    def get_requirements(self) -> dict:
        return {}

    def _get_catalog(self) -> pd.DataFrame:
        catalog = pd.read_csv(self.data_path)
        return catalog

    def _get_rate_fn(self, **kwargs: dict) -> Callable:
        """Returns a callable rate function that takes each of 'columns' as kwargs.
        """
        raise NotImplementedError

    def _get_n_expected(self, **kwargs: dict) -> float:
        """Computes and returns the integral of the rate function
        """
        raise NotImplementedError

    def logp(self, **params_values: dict) -> float:
        rate_fn = self._get_rate_fn(**params_values)
        n_expected = self._get_n_expected(**params_values)
        return self.data.loglike(rate_fn, n_expected)
