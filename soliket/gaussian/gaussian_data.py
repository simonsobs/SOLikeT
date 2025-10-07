from collections.abc import Sequence
from typing import Optional

import numpy as np
import sacc
from cobaya.functions import chi_squared


class GaussianData:
    """
    Named multivariate gaussian data
    """

    name: str  # name identifier for the data
    x: Sequence  # labels for each data point
    y: np.ndarray  # data point values
    cov: np.ndarray  # covariance matrix
    inv_cov: np.ndarray  # inverse covariance matrix
    ncovsims: int | None  # number of simulations used to estimate covariance
    indices: np.ndarray | None  # boolean array to trim cross-cov with selected bandpowers

    _fast_chi_squared = staticmethod(chi_squared)

    def __init__(
        self,
        name,
        x: Sequence,
        y: Sequence[float],
        cov: np.ndarray,
        ncovsims: int | None = None,
        indices: np.ndarray | None = None,
    ):
        self.name = str(name)
        self.ncovsims = ncovsims
        self.indices = (
            indices
            if indices is not None and not all(indices)
            else np.ones(len(x), dtype=bool)
        )

        if not (len(x) == len(y) and cov.shape == (len(x), len(x))):
            raise ValueError(
                f"Incompatible shapes! x={len(x)}, y={len(y)}, \
                               cov={cov.shape}"
            )

        self.x: Sequence[float] = x
        self.y: np.ndarray = np.ascontiguousarray(y)
        self.cov: np.ndarray = cov
        # self.eigenevalues = np.linalg.eigvalsh(cov)
        # if self.eigenevalues.min() <= 0:
        #    print(self.eigenevalues)
        #    raise ValueError("Covariance is not positive definite!")

        self.inv_cov: np.ndarray = np.linalg.inv(self.cov)
        if ncovsims is not None:
            hartlap_factor = (self.ncovsims - len(x) - 2) / (self.ncovsims - 1)
            self.inv_cov *= hartlap_factor
        # log_det = np.log(self.eigenevalues).sum()
        sign_log_det, log_det = np.linalg.slogdet(self.cov)
        if sign_log_det != 1:
            raise ValueError(
                f"Negative or zero determinant: \
                               sign(det)={sign_log_det}"
            )
        self.norm_const = -(np.log(2 * np.pi) * len(x) + log_det) / 2

    def __len__(self) -> int:
        return len(self.x)

    def loglike(self, theory: np.ndarray) -> float:
        delta = self.y - theory
        return -0.5 * self._fast_chi_squared(self.inv_cov, delta) + self.norm_const


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

class MultiGaussianData(GaussianData):
    """

    Parameters
    ----------
    data_list : list
        List of Data objects

    cross_covs : dictionary
        Cross-covariances, keyed by (name1, name2) tuples.
    """

    def __init__(
        self,
        data_list: list[GaussianData],
        cross_covs: dict[tuple[str, str], np.ndarray] | None = None,
    ):
        if cross_covs is None:
            cross_covs = {}

        self.cross_covs = {}

        # Ensure all cross-covs are proper shape, and fill with zeros if not present
        for d1 in data_list:
            for d2 in data_list:
                key = (d1.name, d2.name)

                if d1 == d2:
                    # cross_covs[key] = d1.cov
                    self.cross_covs[key] = d1.cov
                    continue

                rev_key = (d2.name, d1.name)

                if key not in cross_covs and rev_key not in cross_covs:
                    self.cross_covs[key] = np.zeros((len(d1), len(d2)))
                elif key in cross_covs:
                    self.cross_covs[key] = cross_covs[key][d1.indices, :][:, d2.indices]
                    if not self.cross_covs[key].shape == (len(d1), len(d2)):
                        raise ValueError(
                            f"Cross-covariance (for {d1.name} x {d2.name}) \
                              has wrong shape: {self.cross_covs[key].shape} \
                              instead of {len(d1)} x {len(d2)}!"
                        )
                    self.cross_covs[rev_key] = self.cross_covs[key].T

        self.data_list: list[GaussianData] = data_list
        self.lengths: list[int] = [len(d) for d in data_list]
        self.names: list[str] = [d.name for d in data_list]

        self._data: np.ndarray | None = None

    @property
    def data(self) -> GaussianData:
        if self._data is None:
            self._assemble_data()
        return self._data

    def loglike(self, theory: np.ndarray) -> float:
        return self.data.loglike(theory)

    @property
    def name(self) -> str:
        return self.data.name

    @property
    def inv_cov(self) -> np.ndarray:
        return self.data.inv_cov

    @property
    def cov(self) -> np.ndarray:
        return self.data.cov

    @property
    def norm_const(self) -> float:
        return self.data.norm_const

    @property
    def labels(self) -> list[str]:
        return [
            x
            for y in [[name] * len(d) for name, d in zip(self.names, self.data_list)]
            for x in y
        ]

    def _index_range(self, name: str) -> tuple[int, int]:
        if name not in self.names:
            raise ValueError(f"{name} not in {self.names}!")

        i0 = 0
        for n, length in zip(self.names, self.lengths):
            if n == name:
                i1 = i0 + length
                break
            i0 += length
        return i0, i1

    def _slice(self, *names: str) -> slice:
        if isinstance(names, str):
            names = [names]

        return np.s_[tuple(slice(*self._index_range(n)) for n in names)]

    def _assemble_data(self):
        x = np.concatenate([d.x for d in self.data_list])
        y = np.concatenate([d.y for d in self.data_list])

        N = sum([len(d) for d in self.data_list])

        cov = np.zeros((N, N))
        for n1 in self.names:
            for n2 in self.names:
                cov[self._slice(n1, n2)] = self.cross_covs[(n1, n2)]

        self._data = GaussianData(" + ".join(self.names), x, y, cov)

    def plot_cov(self, **kwargs):
        import matplotlib.pyplot as plt

        labels = [
            f"{label}: {value:.2f}" for label, value in zip(self.labels, self.data.x)
        ]

        x_indices = np.arange(len(labels) + 1)
        y_indices = np.arange(len(labels) + 1)

        _, ax = plt.subplots(figsize=(10, 8))
        heatmap = ax.pcolormesh(
            x_indices, y_indices, self.cov, cmap="viridis", shading="auto"
        )

        ax.set_xticks(x_indices[:-1] + 0.5)
        ax.set_yticks(y_indices[:-1] + 0.5)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)

        ax.invert_yaxis()

        plt.colorbar(heatmap, ax=ax)

        plt.show()

        return heatmap
