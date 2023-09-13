import numpy as np
import pandas as pd
import time


def poisson_logpdf(n_expected, catalog, columns, rate_fn, name="unbinned"):
    """Computes log-likelihood of data under poisson process model

    rate_fn returns the *observed rate* as a function of self.columns
    (must be able to take all of self.columns as keywords

    n_expected is predicted total number
    """
    start = time.time()

    rate_densities = np.array(rate_fn(**{c: catalog[c].values for c in columns}))
    assert np.all(np.isfinite(rate_densities))

    elapsed = time.time() - start
    print("\r ::: rate density calculation took {:.3f} seconds.".format(elapsed))

    loglike = -n_expected + np.nansum(np.log(rate_densities[np.nonzero(rate_densities)]))

    print("\r ::: 2D ln likelihood = ", loglike)
    # print("rates:",np.shape(rate_densities),rate_densities)

    return loglike



class PoissonData:
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

    def loglike(self, rate_fn, n_expected):
        return poisson_logpdf(n_expected, self.catalog, self.columns, rate_fn, name=self.name)
