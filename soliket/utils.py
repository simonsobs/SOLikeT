r"""
.. module:: utils

:Synopsis: Compilation of some useful classes and functions for use in SOLikeT.

"""

from importlib import import_module

import numpy as np
from cobaya.likelihood import Likelihood
from cobaya.likelihoods.one import one
from scipy.stats import binned_statistic as binnedstat


def binner(
    ell: np.ndarray, cl_values: np.ndarray, bin_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Simple function intended for binning :math:`\ell`-by-:math:`\ell` data into
    band powers with a top hat window function.

    Note that the centers are computed as :math:`0.5({\rm LHE}+{\rm RHE})`,
    where :math:`{\rm LHE}` and :math:`{\rm RHE}` are the bin edges.
    While this is ok for plotting purposes, the user may need
    to recompute the bin center in case of integer ``ell``
    if the correct baricenter is needed.

    :param ell: Axis along which to bin
    :param cl_values: Values to be binned
    :param bin_edges: The edges of the bins. Note that all but the last bin
                      are open to the right. The last bin is closed.

    :return: The centers of the bins and the average of ``cl_values`` within the bins.
    """
    x = ell.copy()
    y = cl_values.copy()
    cents = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_edges_min = bin_edges.min()
    bin_edges_max = bin_edges.max()
    y[x < bin_edges_min] = 0
    y[x > bin_edges_max] = 0
    bin_means = binnedstat(x, y, bins=bin_edges, statistic=np.nanmean)[0]
    return cents, bin_means


def get_likelihood(name: str, options: dict | None = None) -> Likelihood:
    parts = name.split(".")
    module = import_module(".".join(parts[:-1]))
    t = getattr(module, parts[-1])
    if not issubclass(t, Likelihood):
        raise ValueError(f"{name} is not a Likelihood!")
    if options is None:
        options = {}
    return t(options)


class OneWithCls(one):
    r"""
    Extension of
    `cobaya.likelihoods.one
    <https://cobaya.readthedocs.io/en/latest/likelihood_one.html>`_
    which creates a dummy :math:`C_\ell` requirements dictionary with an
    :math:`\ell_{\rm max}` of 1000 to force computation of ``pp``, ``tt``, ``te``, ``ee``
    and ``bb`` :math:`C_\ell` s.
    """

    lmax = 10000

    def get_requirements(self):
        return {
            "Cl": {
                "pp": self.lmax,
                "tt": self.lmax,
                "te": self.lmax,
                "ee": self.lmax,
                "bb": self.lmax,
            }
        }
