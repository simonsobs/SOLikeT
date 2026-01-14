from collections.abc import Sequence

import numpy as np
import sacc
from cobaya.input import get_default_info, merge_info
from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError
from cobaya.theory import Provider, Theory
from cobaya.tools import recursive_update
from cobaya.typing import empty_dict

from soliket.gaussian.gaussian_data import CrossCov, GaussianData, MultiGaussianData
from soliket.utils import get_likelihood


class GaussianLikelihood(Likelihood):
    """Base class for Gaussian likelihoods in SOLikeT.

    This class provides the infrastructure for computing Gaussian log-likelihoods
    from SACC data files. Subclasses must implement the ``_get_theory()`` method
    to compute the theory prediction for the data vector.

    Parameters
    ----------
    name : str
        Name identifier for the likelihood (default: "Gaussian")
    datapath : str
        Path to the SACC file containing data and covariance
    use_spectra : str or list
        Which spectra to use. Either "all" or a list of tracer pairs
        like ``[("tracer1", "tracer2")]``
    ncovsims : int, optional
        Number of simulations used to estimate covariance. If provided,
        applies the Hartlap correction factor to the inverse covariance.

    Attributes
    ----------
    data : GaussianData
        The assembled Gaussian data object with covariance
    sacc_data : sacc.Sacc
        The loaded SACC data object
    x : np.ndarray
        The bin centers (ell values)
    y : np.ndarray
        The data vector
    cov : np.ndarray
        The covariance matrix

    Examples
    --------
    To create a custom Gaussian likelihood::

        class MyLikelihood(GaussianLikelihood):
            name = "my_likelihood"
            _allowable_tracers = ("cmb_temperature", "cmb_polarization")

            def _get_theory(self, **params):
                # Compute theory prediction
                return theory_vector
    """

    name: str = "Gaussian"
    use_spectra: (
        str | tuple[str, str] | list[tuple[str, str]] | list[list[str, str]] | None
    ) = None
    datapath: str | None = None
    sacc_data: sacc.Sacc | None = None
    ncovsims: int | None = None
    provider: Provider

    _enforce_types: bool = True
    _allowable_tracers: tuple[str] | None = None

    def initialize(self):
        self.log.info(f"Initialising {self.name}...")

        self._check_use_spectra()

        if self.datapath is None and self.sacc_data is None:
            raise LoggedError(
                self.log,
                "You must provide either datapath or sacc_data!",
            )
        self.sacc_data = self._get_sacc_data()

        if self._allowable_tracers is None:
            raise LoggedError(
                self.log,
                "You must set _allowable_tracers in the subclass of GaussianLikelihood!",
            )
        self._check_tracers()
        self.tracer_comb = self.sacc_data.get_tracer_combinations()[0]

        self.data = self._get_gauss_data()

    def _check_use_spectra(self):
        if self.use_spectra is None:
            raise LoggedError(self.log, "You must provide use_spectra!")
        elif isinstance(self.use_spectra, str):
            assert self.use_spectra == "all", "The only allowed string is 'all'!"
        elif isinstance(self.use_spectra, tuple):
            self.use_spectra = [self.use_spectra]
        elif isinstance(self.use_spectra, list):
            for item in self.use_spectra:
                if isinstance(item, list):
                    self.use_spectra[self.use_spectra.index(item)] = tuple(item)
                elif not isinstance(item, tuple) or len(item) != 2:
                    raise LoggedError(
                        self.log,
                        "Each item in `use_spectra` list must "
                        "be a tuple of two tracer names!",
                    )

    def _get_sacc_data(self, **params_values):
        if self.sacc_data is not None:
            self.log.warning(
                "You have provided sacc_data directly, so datapath will be ignored!"
            )
        else:
            self.log.info(f"Loading data from {self.datapath}...")
            sacc_data = sacc.Sacc.load_fits(self.datapath)

        if self.use_spectra == "all":
            pass
        else:
            for tracer_comb in sacc_data.get_tracer_combinations():
                if tracer_comb not in self.use_spectra:
                    sacc_data.remove_selection(tracers=tracer_comb)
        tracer_combs = sacc_data.get_tracer_combinations()
        assert tracer_combs != [], "No tracer was found!"
        return sacc_data

    def _get_gauss_data(self, **params_values):
        self.x = self._construct_ell_bins()
        self.y = self.sacc_data.mean
        self.cov = self.sacc_data.covariance.covmat

        data = GaussianData(self.name, self.x, self.y, self.cov, self.ncovsims)
        return data

    def _check_tracers(self):
        for tracer_comb in self.sacc_data.get_tracer_combinations():
            assert len(tracer_comb) == 2, "Only auto- and cross-spectra are supported!"
            for tracer in tracer_comb:
                if self.sacc_data.tracers[tracer].quantity not in self._allowable_tracers:
                    raise LoggedError(
                        self.log,
                        (
                            f"You have tried to use a "
                            f"{self.sacc_data.tracers[tracer].quantity} tracer in "
                            f"{self.__class__.__name__}, which only allows "
                            f"{self._allowable_tracers}. Please check your "
                            "tracer selection in the ini file."
                        ),
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
        bpw_idx = self.sacc_data.indices(data_type="cl_00", tracers=tracer_comb)
        bpw = self.sacc_data.get_bandpower_windows(bpw_idx)
        ells_theory = bpw.values
        ells_theory = np.asarray(ells_theory, dtype=int)
        w_bins = bpw.weight.T

        return ells_theory, w_bins

    def _get_theory(self, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def logp(self, **params_values) -> float:
        theory = self._get_theory(**params_values)
        return self.data.loglike(theory)


class MultiGaussianLikelihood(GaussianLikelihood):
    """A likelihood combining multiple Gaussian likelihoods with cross-covariances.

    This class enables joint analysis of multiple datasets by combining their
    data vectors and covariance matrices. Cross-covariances between datasets
    can be specified via a ``CrossCov`` object stored in SACC format.

    Parameters
    ----------
    components : list of str
        List of likelihood class names to combine, e.g.,
        ``["soliket.mflike.MFLike", "soliket.lensing.LensingLikelihood"]``
    options : list of dict
        Configuration options for each component likelihood. Each dict should
        contain at minimum ``datapath`` and any other required parameters.
    cross_cov_path : str, optional
        Path to a SACC file containing cross-covariances between components.
        If not provided, components are assumed independent (zero cross-covariance).

    Attributes
    ----------
    likelihoods : list of Likelihood
        The instantiated component likelihoods
    cross_cov : CrossCov or None
        The loaded cross-covariance container
    data : MultiGaussianData
        The combined data object with joint covariance

    Examples
    --------
    YAML configuration::

        likelihood:
          soliket.MultiGaussianLikelihood:
            components:
              - soliket.mflike.MFLike
              - soliket.lensing.LensingLikelihood
            options:
              - datapath: /path/to/mflike.fits
                use_spectra: all
              - datapath: /path/to/lensing.fits
            cross_cov_path: /path/to/cross_cov.fits

    Python usage::

        from soliket import MultiGaussianLikelihood

        info = {
            "components": ["soliket.mflike.MFLike", "soliket.lensing.LensingLikelihood"],
            "options": [
                {"datapath": "mflike.fits", "use_spectra": "all"},
                {"datapath": "lensing.fits"},
            ],
            "cross_cov_path": "cross_cov.fits",
        }
        like = MultiGaussianLikelihood(info)
    """

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
