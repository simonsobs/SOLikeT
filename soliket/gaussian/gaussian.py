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
    use_spectra: str | tuple[str, str] | list[tuple[str, str]] | None = None
    datapath: str | None = None
    sacc_data: sacc.Sacc | None = None
    ncovsims: int | None = None
    provider: Provider

    _enforce_types: bool = True
    _allowable_tracers: tuple[str] | None = None

    def initialize(self):
        self.log.info(f"Initialising {self.name}...")

        if self.use_spectra is None:
            raise LoggedError(self.log, "You must provide use_spectra!")
        elif isinstance(self.use_spectra, str):
            assert self.use_spectra == "all", "The only allowed string is 'all'!"
        elif isinstance(self.use_spectra, tuple):
            self.use_spectra = [self.use_spectra]

        if self.datapath is None:
            if self.sacc_data is None:
                raise LoggedError(
                    self.log,
                    "You must provide either datapath or sacc_data!",
                )
        else:
            self.sacc_data = self._get_sacc_data()

        if self._allowable_tracers is None:
            raise LoggedError(
                self.log,
                "You must set _allowable_tracers in the subclass of GaussianLikelihood!",
            )
        self._check_tracers()
        self.tracer_comb = self.sacc_data.get_tracer_combinations()[0]

        self.data = self._get_gauss_data()

    def _get_sacc_data(self, **params_values):
        if self.sacc_data is not None:
            self.log.warning(
                "You have provided sacc_data directly, so datapath will be ignored!"
            )
        else:
            print(f"Loading data from {self.datapath}...")
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


class CrossCov(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metadata = {}

    def add_metadata(
        self,
        key: tuple[str],
        tracers: tuple[tuple[str]],
        data_types: tuple[str],
        tracer_info: dict[str, dict[str, str | int]] = None,
    ):
        """Store metadata for cross-covariance entries.

        Parameters:
        -----------
        key : tuple[str]
            Component identifier key
        tracers : tuple[tuple[str]]
            Tracer pairs for each component
        data_types : tuple[str]
            Data types (e.g., "cl_00", "cl_22")
        tracer_info : dict[str, dict[str, str | int]]
            Dictionary mapping tracer names to their properties:
            {tracer_name: {"name": str, "quantity": str, "spin": int}}
        """
        self._metadata[key] = {
            "tracers": tracers,
            "data_types": data_types,
            "tracer_info": tracer_info or {},
        }

    def save(self, path: str):
        assert path.endswith((".fits", ".sacc")), "Only 'sacc' files are supported!"

        cross_sacc = sacc.Sacc()

        # Collect all unique tracers from metadata
        all_tracers = set()
        tracer_info_dict = {}

        for metadata in self._metadata.values():
            for tracer_pair in metadata["tracers"]:
                all_tracers.update(tracer_pair)

            # Collect tracer info from metadata
            if "tracer_info" in metadata:
                tracer_info_dict.update(metadata["tracer_info"])

        # Add tracers using the stored metadata
        for tracer_name in sorted(all_tracers):
            if tracer_name in tracer_info_dict:
                info = tracer_info_dict[tracer_name]
                cross_sacc.add_tracer(
                    "misc", info["name"], quantity=info["quantity"], spin=info["spin"]
                )
            else:
                raise ValueError(
                    f"No tracer info provided for '{tracer_name}'. "
                    "Please add tracer info using add_metadata() "
                    "with tracer_info parameter."
                )

        # Add minimal data points to establish SACC structure
        # We need actual data points before we can set covariance
        for key in self.keys():
            tracer_1, tracer_2 = key
            if tracer_1 in tracer_info_dict and tracer_2 in tracer_info_dict:
                info_1 = tracer_info_dict[tracer_1]
                info_2 = tracer_info_dict[tracer_2]
                data_type = f"cl_{info_1['spin']}{info_2['spin']}"

                matrix = self[key]
                n_ell = matrix.shape[0]
                ells = np.arange(2, 2 + n_ell)  # Start from ell=2
                cls = np.zeros(n_ell)  # Dummy Cl values

                cross_sacc.add_ell_cl(data_type, tracer_1, tracer_2, ells, cls)

        # Build full covariance matrix from cross-covariance blocks
        full_cov = self._build_full_covariance_matrix(cross_sacc)
        cross_sacc.add_covariance(full_cov)

        cross_sacc.save_fits(path, overwrite=True)

    @classmethod
    def load(cls, path: str | None) -> Optional["CrossCov"]:
        """Load cross-covariances from SACC format."""
        if path is None:
            return None

        if not path.endswith((".fits", ".sacc")):
            raise ValueError("Only .fits or .sacc files are supported for CrossCov!")

        cross_sacc = sacc.Sacc.load_fits(path)
        cross_cov = cls()

        # Rebuild tracer_info from SACC tracers
        tracer_info_dict = {}
        for tracer_name, tracer in cross_sacc.tracers.items():
            tracer_info_dict[tracer_name] = {
                "name": tracer_name,
                "quantity": tracer.quantity,
                "spin": getattr(tracer, "spin", 0),  # Default to 0 if no spin attribute
            }

        # Extract cross-covariance blocks from the full covariance matrix
        if hasattr(cross_sacc, "covariance") and cross_sacc.covariance is not None:
            full_cov = cross_sacc.covariance.covmat
            tracer_combinations = cross_sacc.get_tracer_combinations()

            # Create mapping from tracer combinations to data indices
            tracer_to_indices = {}
            for tracer_comb in tracer_combinations:
                indices = cross_sacc.indices(tracers=tracer_comb)
                tracer_to_indices[tracer_comb] = indices

            # Extract cross-covariance blocks
            for tracer_comb_i in tracer_combinations:
                indices_i = tracer_to_indices[tracer_comb_i]

                for tracer_comb_j in tracer_combinations:
                    indices_j = tracer_to_indices[tracer_comb_j]

                    # Extract the covariance block
                    cov_block = full_cov[np.ix_(indices_i, indices_j)]

                    # Store significant cross-covariances (skip diagonal auto-cov)
                    if tracer_comb_i != tracer_comb_j and not np.allclose(
                        cov_block, 0, atol=1e-12
                    ):
                        key = (tracer_comb_i[0], tracer_comb_j[0])
                        cross_cov[key] = cov_block

                        # Add metadata
                        spin_i0 = getattr(cross_sacc.tracers[tracer_comb_i[0]], "spin", 0)
                        spin_i1 = getattr(cross_sacc.tracers[tracer_comb_i[1]], "spin", 0)
                        spin_j0 = getattr(cross_sacc.tracers[tracer_comb_j[0]], "spin", 0)
                        spin_j1 = getattr(cross_sacc.tracers[tracer_comb_j[1]], "spin", 0)

                        cross_cov._metadata[key] = {
                            "tracers": (tracer_comb_i, tracer_comb_j),
                            "data_types": (
                                f"cl_{spin_i0}{spin_i1}",
                                f"cl_{spin_j0}{spin_j1}",
                            ),
                            "tracer_info": tracer_info_dict,
                        }

        return cross_cov

    def _build_full_covariance_matrix(self, sacc_obj: sacc.Sacc) -> np.ndarray:
        """Build the full covariance matrix from cross-covariance blocks."""
        tracer_combinations = sacc_obj.get_tracer_combinations()
        n_data = len(sacc_obj.mean)
        full_cov = np.zeros((n_data, n_data))

        # Create mapping from tracer combinations to data indices
        tracer_to_indices = {}
        current_idx = 0

        for tracer_comb in tracer_combinations:
            indices = sacc_obj.indices(tracers=tracer_comb)
            tracer_to_indices[tracer_comb] = indices
            current_idx += len(indices)

        # Fill the covariance matrix with cross-covariance blocks
        for tracer_comb_i in tracer_combinations:
            indices_i = tracer_to_indices[tracer_comb_i]

            for tracer_comb_j in tracer_combinations:
                indices_j = tracer_to_indices[tracer_comb_j]

                # Find the appropriate cross-covariance block
                cross_cov_block = None

                if tracer_comb_i == tracer_comb_j:
                    # Diagonal block - use identity if no self-covariance available
                    if tracer_comb_i in self:
                        cross_cov_block = self[tracer_comb_i]
                    else:
                        cross_cov_block = np.eye(len(indices_i))
                else:
                    # Off-diagonal block
                    if tracer_comb_i in self and tracer_comb_j in self:
                        # Both tracers have individual data, check for cross-cov
                        cross_key = (tracer_comb_i[0], tracer_comb_j[0])
                        rev_cross_key = (tracer_comb_j[0], tracer_comb_i[0])

                        if cross_key in self:
                            cross_cov_block = self[cross_key]
                        elif rev_cross_key in self:
                            cross_cov_block = self[rev_cross_key].T
                        else:
                            cross_cov_block = np.zeros((len(indices_i), len(indices_j)))
                    else:
                        cross_cov_block = np.zeros((len(indices_i), len(indices_j)))

                # Place the block in the full covariance matrix
                if cross_cov_block is not None:
                    for idx_i, global_i in enumerate(indices_i):
                        for idx_j, global_j in enumerate(indices_j):
                            if (
                                idx_i < cross_cov_block.shape[0]
                                and idx_j < cross_cov_block.shape[1]
                            ):
                                full_cov[global_i, global_j] = cross_cov_block[
                                    idx_i, idx_j
                                ]

        return full_cov


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
