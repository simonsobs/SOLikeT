import json
from collections.abc import Sequence
from typing import Optional

import numpy as np
import sacc
from cobaya.functions import chi_squared


class GaussianData:
    """Container for named multivariate Gaussian data.

    Stores a data vector with its covariance matrix and provides methods
    for computing the Gaussian log-likelihood.

    Parameters
    ----------
    name : str
        Name identifier for the data
    x : Sequence
        Labels or coordinates for each data point (e.g., ell values)
    y : Sequence[float]
        The data vector values
    cov : np.ndarray
        Covariance matrix with shape (n, n) where n = len(x)
    ncovsims : int, optional
        Number of simulations used to estimate covariance. If provided,
        applies the Hartlap correction factor to the inverse covariance.
    indices : np.ndarray, optional
        Boolean array for trimming cross-covariances when scale cuts are applied

    Attributes
    ----------
    inv_cov : np.ndarray
        Inverse covariance matrix (with Hartlap correction if applicable)
    norm_const : float
        Normalization constant for the Gaussian likelihood

    Raises
    ------
    ValueError
        If dimensions of x, y, and cov are incompatible
        If covariance matrix has non-positive determinant
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
        """Compute the Gaussian log-likelihood.

        Parameters
        ----------
        theory : np.ndarray
            Theory prediction vector with same length as data

        Returns
        -------
        float
            Log-likelihood value including normalization constant
        """
        delta = self.y - theory
        return -0.5 * self._fast_chi_squared(self.inv_cov, delta) + self.norm_const


class CrossCov(dict):
    """Cross-covariance container for multi-component Gaussian likelihoods.

    Stores cross-covariances between named components (e.g., "mflike", "lensing")
    and optionally the full covariances for each component. Supports saving and
    loading in SACC format for persistence.

    The dictionary keys are tuples of component names, e.g., ("mflike", "lensing").
    Values are the corresponding covariance matrices.

    For the full joint covariance, use:

    - Diagonal blocks: ``(name, name)`` -> auto-covariance matrix
    - Off-diagonal blocks: ``(name1, name2)`` -> cross-covariance matrix

    Examples
    --------
    **Mode 1: Full covariance specification**

    Use ``add_component()`` for auto-covariances and ``add_cross_covariance()``
    for off-diagonal blocks::

        cross_cov = CrossCov()
        cross_cov.add_component("mflike", mflike_cov)
        cross_cov.add_component("lensing", lensing_cov)
        cross_cov.add_cross_covariance("mflike", "lensing", cross_block)
        cross_cov.save("cross_cov.fits")

    **Mode 2: Cross-covariance only**

    If auto-covariances will come from individual likelihoods::

        cross_cov = CrossCov()
        cross_cov.add_cross_covariance("mflike", "lensing", cross_block)
        cross_cov.save("cross_cov.fits")

    **Loading**::

        cross_cov = CrossCov.load("cross_cov.fits")
        block = cross_cov[("mflike", "lensing")]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metadata = {}
        self._component_info: dict[str, dict] = {}

    def add_component(
        self,
        name: str,
        cov: np.ndarray,
    ):
        """Add a component with its full covariance.

        Parameters
        ----------
        name : str
            Component name (e.g., "mflike", "kk")
        cov : np.ndarray
            Full covariance matrix for this component
        """
        if isinstance(cov, dict):
            raise TypeError(
                f"cov must be a numpy array, not a dict. Got: {type(cov)}"
            )
        cov_array = np.asarray(cov)
        self._component_info[name] = {
            "size": cov_array.shape[0],
            "cov": cov_array,
        }
        # Also store the auto-covariance in the dict
        self[(name, name)] = cov_array

    def add_cross_covariance(
        self,
        name1: str,
        name2: str,
        cross_cov: np.ndarray,
    ):
        """Add cross-covariance between two components.

        Parameters
        ----------
        name1 : str
            First component name
        name2 : str
            Second component name
        cross_cov : np.ndarray
            Cross-covariance matrix with shape (n1, n2)
        """
        self[(name1, name2)] = np.asarray(cross_cov)
        # Also store the transpose for convenience
        self[(name2, name1)] = np.asarray(cross_cov).T

    @property
    def component_names(self) -> list[str]:
        """Get ordered list of component names."""
        return list(self._component_info.keys())

    def _infer_component_info(self):
        """Infer component sizes from stored covariance blocks.

        This is called when save() is invoked without explicit add_component() calls.
        Sizes are inferred from the shapes of cross-covariance matrices.
        """
        sizes: dict[str, int] = {}

        for key, cov in self.items():
            name1, name2 = key
            n1, n2 = cov.shape

            if name1 in sizes:
                if sizes[name1] != n1:
                    raise ValueError(
                        f"Inconsistent sizes for component '{name1}': "
                        f"{sizes[name1]} vs {n1}"
                    )
            else:
                sizes[name1] = n1

            if name2 in sizes:
                if sizes[name2] != n2:
                    raise ValueError(
                        f"Inconsistent sizes for component '{name2}': "
                        f"{sizes[name2]} vs {n2}"
                    )
            else:
                sizes[name2] = n2

        # Populate _component_info with inferred sizes
        for name, size in sizes.items():
            self._component_info[name] = {
                "size": size,
                "cov": self.get((name, name)),  # May be None if only cross-covs
            }

    def save(self, path: str):
        """Save cross-covariance to SACC format.

        The SACC file will contain:
        - A misc tracer for each component
        - Dummy data points to establish the data vector structure
        - The full joint covariance matrix

        Parameters
        ----------
        path : str
            Output path (must end with .fits or .sacc)
        """
        if not path.endswith((".fits", ".sacc")):
            raise ValueError("Only .fits or .sacc files are supported!")

        # If no components were added explicitly, infer sizes from cross-covariances
        if not self._component_info:
            self._infer_component_info()

        cross_sacc = sacc.Sacc()

        # Add metadata about component order (as JSON string for FITS compatibility)
        cross_sacc.metadata["component_names"] = json.dumps(self.component_names)

        # Add a misc tracer for each component
        for name in self.component_names:
            cross_sacc.add_tracer("misc", name, quantity="generic", spin=0)

        # Add dummy data points to establish SACC structure
        # (SACC requires data points before covariance can be added)
        # The actual data values are not meaningful - only the covariance matters
        for name in self.component_names:
            n_points = self._component_info[name]["size"]

            for i in range(n_points):
                cross_sacc.add_data_point(
                    "generic", (name, name), 0.0, ell=float(i)
                )

        # Build and add the full joint covariance matrix
        full_cov = self._build_full_covariance()
        cross_sacc.add_covariance(full_cov)

        cross_sacc.save_fits(path, overwrite=True)

    def _build_full_covariance(self) -> np.ndarray:
        """Build the full joint covariance matrix from stored blocks."""
        names = self.component_names
        sizes = [self._component_info[name]["size"] for name in names]
        total_size = sum(sizes)

        full_cov = np.zeros((total_size, total_size))

        # Fill in blocks
        row_start = 0
        for i, name_i in enumerate(names):
            col_start = 0
            for j, name_j in enumerate(names):
                key = (name_i, name_j)
                if key in self:
                    block = np.asarray(self[key])
                    full_cov[
                        row_start : row_start + sizes[i],
                        col_start : col_start + sizes[j],
                    ] = block
                col_start += sizes[j]
            row_start += sizes[i]

        return full_cov

    @classmethod
    def load(cls, path: str | None) -> Optional["CrossCov"]:
        """Load cross-covariance from SACC format.

        Parameters
        ----------
        path : str or None
            Path to SACC file. If None, returns None.

        Returns
        -------
        CrossCov or None
            Loaded cross-covariance object, or None if path is None.
        """
        if path is None:
            return None

        if not path.endswith((".fits", ".sacc")):
            raise ValueError("Only .fits or .sacc files are supported!")

        cross_sacc = sacc.Sacc.load_fits(path)
        cross_cov = cls()

        # Get component names from metadata or infer from tracers
        if "component_names" in cross_sacc.metadata:
            # Parse JSON string back to list
            component_names = json.loads(cross_sacc.metadata["component_names"])
        else:
            # Infer from tracer names
            component_names = list(cross_sacc.tracers.keys())

        # Extract indices for each component
        component_indices = {}
        for name in component_names:
            indices = cross_sacc.indices(tracers=(name, name))
            component_indices[name] = indices

            # Store component info (cov will be extracted below)
            cross_cov._component_info[name] = {
                "size": len(indices),
                "cov": None,  # Will be filled below
            }

        # Extract covariance blocks
        if cross_sacc.covariance is not None:
            full_cov = cross_sacc.covariance.covmat

            for name_i in component_names:
                indices_i = component_indices[name_i]

                for name_j in component_names:
                    indices_j = component_indices[name_j]

                    # Extract the covariance block
                    cov_block = full_cov[np.ix_(indices_i, indices_j)]
                    cross_cov[(name_i, name_j)] = cov_block

                    # Update auto-covariance in component_info
                    if name_i == name_j:
                        cross_cov._component_info[name_i]["cov"] = cov_block

        return cross_cov

    # Keep old methods for backwards compatibility with existing metadata approach
    def add_metadata(
        self,
        key: tuple[str],
        tracers: tuple[tuple[str]],
        data_types: tuple[str],
        tracer_info: dict[str, dict[str, str | int]] = None,
    ):
        """Store metadata for cross-covariance entries (legacy method).

        Parameters
        ----------
        key : tuple[str]
            Component identifier key
        tracers : tuple[tuple[str]]
            Tracer pairs for each component
        data_types : tuple[str]
            Data types (e.g., "cl_00", "cl_22")
        tracer_info : dict[str, dict[str, str | int]]
            Dictionary mapping tracer names to their properties
        """
        self._metadata[key] = {
            "tracers": tracers,
            "data_types": data_types,
            "tracer_info": tracer_info or {},
        }

class MultiGaussianData(GaussianData):
    """Combined Gaussian data from multiple components with cross-covariances.

    Assembles multiple ``GaussianData`` objects into a single joint data vector
    with a combined covariance matrix that includes both auto-covariances and
    cross-covariances between components.

    Parameters
    ----------
    data_list : list of GaussianData
        Individual data objects to combine
    cross_covs : CrossCov, optional
        Cross-covariance container. If None, components are assumed independent.
        Auto-covariances can come from either the CrossCov or the individual
        GaussianData objects (individual data takes precedence if CrossCov
        doesn't contain auto-covariance for a component).

    Attributes
    ----------
    data_list : list of GaussianData
        The original individual data objects
    names : list of str
        Names of all components
    lengths : list of int
        Data vector lengths for each component
    labels : list of str
        Component name for each element in the combined data vector

    Examples
    --------
    Combining two datasets with cross-covariance::

        data1 = GaussianData("mflike", x1, y1, cov1)
        data2 = GaussianData("lensing", x2, y2, cov2)

        cross_cov = CrossCov()
        cross_cov.add_cross_covariance("mflike", "lensing", cross_block)

        multi_data = MultiGaussianData([data1, data2], cross_cov)

        # Access combined properties
        print(multi_data.cov.shape)  # (n1 + n2, n1 + n2)
        loglike = multi_data.loglike(theory_vector)
    """

    def __init__(
        self,
        data_list: list[GaussianData],
        cross_covs: CrossCov | None = None,
    ):
        if cross_covs is None:
            cross_covs = CrossCov()

        self.cross_covs = {}

        # Build covariance blocks: use CrossCov if available, otherwise use defaults
        for d1 in data_list:
            for d2 in data_list:
                key = (d1.name, d2.name)
                rev_key = (d2.name, d1.name)

                if d1 == d2:
                    # Auto-covariance: prefer CrossCov, fallback to individual likelihood
                    if key in cross_covs:
                        cov_block = cross_covs[key]
                        # Only trim if not already the right size
                        if cov_block.shape == (len(d1), len(d1)):
                            self.cross_covs[key] = cov_block
                        else:
                            self.cross_covs[key] = cov_block[d1.indices, :][:, d1.indices]
                    else:
                        # Fallback to individual likelihood's covariance
                        self.cross_covs[key] = d1.cov
                    continue

                # Cross-covariance: use CrossCov if available, otherwise zeros
                if key not in cross_covs and rev_key not in cross_covs:
                    self.cross_covs[key] = np.zeros((len(d1), len(d2)))
                elif key in cross_covs:
                    cov_block = cross_covs[key]
                    # Only trim if not already the right size
                    if cov_block.shape == (len(d1), len(d2)):
                        self.cross_covs[key] = cov_block
                    else:
                        self.cross_covs[key] = cov_block[d1.indices, :][:, d2.indices]
                        if self.cross_covs[key].shape != (len(d1), len(d2)):
                            raise ValueError(
                                f"Cross-covariance (for {d1.name} x {d2.name}) "
                                f"has wrong shape: {self.cross_covs[key].shape} "
                                f"instead of {len(d1)} x {len(d2)}!"
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
