import numpy as np
from typing import Optional, Sequence

from cobaya.likelihood import Likelihood
from cobaya.input import merge_info
from cobaya.tools import recursive_update
#from cobaya.conventions import empty_dict
empty_dict  = 'empty_dict'

from .gaussian_data import GaussianData, MultiGaussianData
from .utils import get_likelihood


class GaussianLikelihood(Likelihood):
    name: str = "Gaussian"
    datapath: Optional[str] = None
    covpath: Optional[str] = None

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

    def _get_theory(self, **kwargs):
        raise NotImplementedError

    def logp(self, **params_values):
        theory = self._get_theory(**params_values)
        return self.data.loglike(theory)


class CrossCov(dict):
    def save(self, path):
        np.savez(path, **{str(k): v for k, v in self.items()})

    @classmethod
    def load(cls, path):
        if path is None:
            return None
        return cls({eval(k): v for k, v in np.load(path).items()})


class MultiGaussianLikelihood(GaussianLikelihood):
    components: Optional[Sequence] = None
    options: Optional[Sequence] = None
    cross_cov_path: Optional[str] = None

    def __init__(self, info=empty_dict, **kwargs):

        if 'components' in info:
            self.likelihoods = [get_likelihood(*kv) for kv in zip(info['components'], info['options'])]

        default_info = merge_info(*[like.get_defaults() for like in self.likelihoods])
        default_info.update(info)

        super().__init__(info=default_info, **kwargs)

    def initialize(self):
        self.cross_cov = CrossCov.load(self.cross_cov_path)

        data_list = [like.data for like in self.likelihoods]
        self.data = MultiGaussianData(data_list, self.cross_cov)

        self.log.info('Initialized.')

    def initialize_with_provider(self, provider):
        for like in self.likelihoods:
            like.initialize_with_provider(provider)
        # super().initialize_with_provider(provider)

    def get_helper_theories(self):
        helpers = {}
        for like in self.likelihoods:
            helpers.update(like.get_helper_theories())

        return helpers

    def _get_theory(self, **kwargs):
        return np.concatenate([like._get_theory(**kwargs) for like in self.likelihoods])

    def get_requirements(self):

        # Reqs with arguments like 'lmax', etc. may have to be carefully treated here to merge
        reqs = {}
        for like in self.likelihoods:
            new_reqs = like.get_requirements()

            # Deal with special cases requiring careful merging
            # Make sure the max of the lmax/union of Cls is taken.
            # (should make a unit test for this)
            if "Cl" in new_reqs and "Cl" in reqs:
                new_cl_spec = new_reqs["Cl"]
                old_cl_spec = reqs["Cl"]
                merged_cl_spec = {}
                all_keys = set(new_cl_spec.keys()).union(set(old_cl_spec.keys()))
                for k in all_keys:
                    new_lmax = new_cl_spec.get(k, 0)
                    old_lmax = old_cl_spec.get(k, 0)
                    merged_cl_spec[k] = max(new_lmax, old_lmax)
                new_reqs["Cl"] = merged_cl_spec

            reqs = recursive_update(reqs, new_reqs)
        return reqs
