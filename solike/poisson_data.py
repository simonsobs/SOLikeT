from inspect import signature

import numpy as np
import pandas as pd


class PoissonData(object):
    """Poisson-process-generated data.

    Parameters
    ----------
    catalog : pd.DataFrame
        Catalog of observed data.
    columns : list
        Columns of catalog relevant for computing poisson rate.
    """

    def __init__(self, name, catalog, columns):
        self.name = str(name)

        self.catalog = pd.DataFrame(catalog)[columns]
        self.columns = columns

    def __len__(self):
        return len(self.catalog)

    def loglike(self, rate_fn, n_expected):
        """Computes log-likelihood of data under poisson process model

        rate_fn returns a rate as a function of self.columns 
        (must be able to take all of self.columns as keywords)

        n_expected is predicted total number
        """
        rate_densities = rate_fn(**{c: self.catalog[c] for c in self.columns})

        return -n_expected + sum(np.log(rate_densities))
