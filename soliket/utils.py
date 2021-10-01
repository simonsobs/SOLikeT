from importlib import import_module

from scipy.stats import binned_statistic as binnedstat
import numpy as np

from cobaya.likelihood import Likelihood
from cobaya.likelihoods.one import one


def binner(ls, cls, bin_edges):
    x = ls.copy()
    y = cls.copy()
    cents = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_edges_min = bin_edges.min()
    bin_edges_max = bin_edges.max()
    y[x < bin_edges_min] = 0
    y[x > bin_edges_max] = 0
    bin_means = binnedstat(x, y, bins=bin_edges, statistic=np.nanmean)[0]
    return cents, bin_means


def get_likelihood(name, options=None):
    parts = name.split(".")
    module = import_module(".".join(parts[:-1]))
    t = getattr(module, parts[-1])
    if not issubclass(t, Likelihood):
        raise ValueError(f"{name} is not a Likelihood!")
    if options is None:
        options = {}
    return t(options)


class OneWithCls(one):
    lmax = 10000

    def get_requirements(self):
        return {"Cl": {"pp": self.lmax,
                       "tt": self.lmax,
                       "te": self.lmax,
                       "ee": self.lmax,
                       "bb": self.lmax, }}
