from collections.abc import Sequence
from typing import Optional

import numpy as np
import sacc
from cobaya.input import get_default_info, merge_info
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError
from cobaya.theory import Provider, Theory
from cobaya.tools import recursive_update
from cobaya.typing import empty_dict

from soliket.gaussian.gaussian_data import GaussianData, MultiGaussianData
from soliket.utils import get_likelihood


class GaussianLikelihood(Likelihood):
    name: str = "Gaussian"
    use_spectra: str | list[tuple[str, str]] | None = None
    datapath: str | None = None
    covpath: str | None = None
    ncovsims: int | None = None
    provider: Provider

    _enforce_types: bool = True
    _allowable_tracers: tuple[str] | None = None

    def initialize(self):
        self.log.info("Initialising.")
        if self.datapath is None:
            raise LoggedError(self.log, "You must provide a datapath!")
        if self.use_spectra is None:
            raise LoggedError(self.log, "You must provide use_spectra!")
        if self._allowable_tracers is None:
            raise LoggedError(
                self.log,
                "You must set _allowable_tracers in the subclass of GaussianLikelihood!",
            )

        self._get_sacc_data()
        self._check_tracers()

        self.data = GaussianData(self.name, self.x, self.y, self.cov, self.ncovsims)

        self.tracer_1, self.tracer_2 = self._allowable_tracers

        self.binning_matrix = self.get_binning((self.tracer_1, self.tracer_2))

    def _get_sacc_data(self, **params_values):
        self.sacc_data = sacc.Sacc.load_fits(self.datapath)

        if self.use_spectra == "all":
            pass
        else:
            for tracer_comb in self.sacc_data.get_tracer_combinations():
                if tracer_comb not in self.use_spectra:
                    self.sacc_data.remove_selection(tracers=tracer_comb)

        self.x = self._construct_ell_bins()
        self.y = self.sacc_data.mean
        self.cov = self.sacc_data.covariance.covmat

        self.data = GaussianData(self.name, self.x, self.y, self.cov, self.ncovsims)

    def _check_tracers(self):
        for tracer_comb in self.sacc_data.get_tracer_combinations():
            assert len(tracer_comb) == 2, "Only auto- and cross-spectra are supported!"
            for tracer in tracer_comb:
                if self.sacc_data.tracers[tracer].quantity not in self._allowable_tracers:
                    raise LoggedError(
                        self.log,
                        f"You have tried to use a \
                        {self.sacc_data.tracers[tracer].quantity} tracer in \
                        {self.__class__.__name__}, which only allows \
                        {self._allowable_tracers}. Please check your \
                        tracer selection in the ini file.",
                    )

    def _construct_ell_bins(self) -> np.ndarray:
        ell_eff = []

        for tracer_comb in self.sacc_data.get_tracer_combinations():
            ind = self.sacc_data.indices(tracers=tracer_comb)
            ell = np.array(self.sacc_data._get_tags_by_index(["ell"], ind)[0])
            ell_eff.append(ell)

        return np.concatenate(ell_eff)

    def _get_data(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x, self.y

    def _get_cov(self) -> np.ndarray:
        return self.cov

    def _get_bin_centers(self) -> np.ndarray:
        return self.x

    def _get_data_spectrum(self) -> np.ndarray:
        return self.y

    def get_binning(self, tracer_comb: tuple) -> tuple[np.ndarray, np.ndarray]:
        bpw_idx = self.sacc_data.indices(tracers=tracer_comb)
        bpw = self.sacc_data.get_bandpower_windows(bpw_idx)
        ells_theory = bpw.values
        ells_theory = np.asarray(ells_theory, dtype=int)
        w_bins = bpw.weight.T

        return ells_theory, w_bins

    def _get_theory(self, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def _get_gauss_data(self):
        return self.data

    def logp(self, **params_values) -> float:
        theory = self._get_theory(**params_values)
        return self.data.loglike(theory)


class CrossCov(dict):
    def save(self, path: str):
        np.savez(path, **{str(k): v for k, v in self.items()})

    @classmethod
    def load(cls, path: str | None) -> Optional["CrossCov"]:
        if path is None:
            return None
        return cls({eval(k): v for k, v in np.load(path).items()})


class MultiGaussianLikelihood(GaussianLikelihood):
    components: Sequence | None = None
    options: Sequence | None = None
    cross_cov_path: str | None = None

    def __init__(self, info=empty_dict, **kwargs):
        if "components" in info:
            self.likelihoods: list[Likelihood] = [
                get_likelihood(*kv) for kv in zip(info["components"], info["options"])
            ]

        default_info = self.get_defaults(input_options=info)
        default_info.update(info)
        default_info = self.get_modified_defaults(default_info, input_options=info)

        super().__init__(info=default_info, **kwargs)

    @classmethod
    def get_defaults(
        cls, return_yaml=False, yaml_expand_defaults=True, input_options=empty_dict
    ):
        default_info = merge_info(
            *[
                get_default_info(like, input_options=info)
                for like, info in zip(
                    input_options["components"], input_options["options"]
                )
            ]
        )

        return default_info

    @classmethod
    def get_modified_defaults(cls, defaults, input_options=empty_dict):
        return defaults

    def initialize(self):
        self.cross_cov: CrossCov | None = CrossCov.load(self.cross_cov_path)

        data_list = [like._get_gauss_data() for like in self.likelihoods]
        self.data = MultiGaussianData(data_list, self.cross_cov)

        self.log.info("Initialized.")

    def initialize_with_provider(self, provider: Provider):  # pragma: no cover
        for like in self.likelihoods:
            like.initialize_with_provider(provider)
        # super().initialize_with_provider(provider)

    def get_helper_theories(self) -> dict[str, Theory]:  # pragma: no cover
        helpers: dict[str, Theory] = {}
        for like in self.likelihoods:
            helpers.update(like.get_helper_theories())

        return helpers

    def _get_theory(self, **kwargs) -> np.ndarray:
        return np.concatenate([like._get_theory(**kwargs) for like in self.likelihoods])

    def get_requirements(self):  # pragma: no cover
        # Reqs with arguments like 'lmax', etc. may have to be carefully treated here to
        # merge
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
