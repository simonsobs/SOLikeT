import numpy as np

from cobaya.likelihood import Likelihood

from .gaussian_data import GaussianData, MultiGaussianData


class GaussianLikelihood(Likelihood):

    class_options = {
        "name": "Gaussian",
        "datapath": None,
        "covpath": None,
    }

    def initialize(self):
        x, y = self._get_data()
        cov = self._get_cov()
        self.data = GaussianData(self.name, x, y, cov)

    def _get_data(self):
        x, y = np.loadtxt(self.datapath, unpack=True)
        return x, y

    def _get_cov(self):
        cov = np.loadtxt(self.covpath)
        return cov

    def _get_theory(self):
        raise NotImplementedError

    def logp(self, **params_values):
        theory = self._get_theory()
        return self.data.norm.logpdf(theory)


class MultiGaussianLikelihood(GaussianLikelihood):
    def __init__(self, likelihoods, cross_cov=None, **kwargs):
        self.likelihoods = likelihoods
        self.cross_cov = cross_cov
        super().__init__(**kwargs)

    def initialize(self):
        data_list = [l.data for l in self.likelihoods]
        self.data = MultiGaussianData(data_list, self.cross_cov)

    def _get_theory(self):
        return np.concatenate([l._get_theory() for l in self.likelihoods])
