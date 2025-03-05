from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from cobaya import functions


class GaussianData:
    """
     Named multivariate gaussian data
    """
    name: str  # name identifier for the data
    x: Sequence  # labels for each data point
    y: np.ndarray  # data point values
    cov: np.ndarray  # covariance matrix
    inv_cov: np.ndarray  # inverse covariance matrix
    ncovsims: Optional[int]  # number of simulations used to estimate covariance

    _fast_chi_squared = staticmethod(functions.chi_squared)

    def __init__(self, name: str, x: Sequence[float], y: Sequence[float], cov: np.ndarray,
                 ncovsims: Optional[int] = None):

        self.name: str = str(name)
        self.ncovsims: Optional[int] = ncovsims

        if not (len(x) == len(y) and cov.shape == (len(x), len(x))):
            raise ValueError(f"Incompatible shapes! x={len(x)}, y={len(y)}, \
                               cov={cov.shape}")

        self.x: Sequence[float] = x
        self.y: np.ndarray = np.ascontiguousarray(y)
        self.cov: np.ndarray = cov
        self.eigenevalues: np.ndarray = np.linalg.eigvalsh(cov)
        if self.eigenevalues.min() <= 0:
            raise ValueError("Covariance is not positive definite!")

        self.inv_cov: np.ndarray = np.linalg.inv(self.cov)
        if ncovsims is not None:
            hartlap_factor = (self.ncovsims - len(x) - 2) / (self.ncovsims - 1)
            self.inv_cov *= hartlap_factor
        log_det = np.log(self.eigenevalues).sum()
        self.norm_const = -(np.log(2 * np.pi) * len(x) + log_det) / 2

    def __len__(self) -> int:
        return len(self.x)

    def loglike(self, theory: np.ndarray) -> float:
        delta = self.y - theory
        return -0.5 * self._fast_chi_squared(self.inv_cov, delta) + self.norm_const


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
        data_list: List[GaussianData],
        cross_covs: Optional[Dict[Tuple[str, str], np.ndarray]] = None,
    ):

        if cross_covs is None:
            cross_covs = {}

        # Ensure all cross-covs are proper shape, and fill with zeros if not present
        for d1 in data_list:
            for d2 in data_list:
                key = (d1.name, d2.name)

                if d1 == d2:
                    cross_covs[key] = d1.cov

                rev_key = (d2.name, d1.name)
                if key in cross_covs:
                    cov = cross_covs[key]
                    if not cov.shape == (len(d1), len(d2)):
                        raise ValueError(
                            f"Cross-covariance (for {d1.name} x {d2.name}) \
                              has wrong shape: {cov.shape}!"
                        )
                elif rev_key in cross_covs:
                    cross_covs[key] = cross_covs[rev_key].T
                else:
                    cross_covs[key] = np.zeros((len(d1), len(d2)))

        self.data_list: List[GaussianData] = data_list
        self.lengths: List[int] = [len(d) for d in data_list]
        self.names: List[str] = [d.name for d in data_list]
        self.cross_covs: Dict[Tuple[str, str], np.ndarray] = cross_covs

        self._data: Optional[np.ndarray] = None

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
    def labels(self) -> List[str]:
        return [
            x
            for y in [[name] * len(d) for name, d in zip(self.names, self.data_list)]
            for x in y
        ]

    def _index_range(self, name: str) -> Tuple[int, int]:
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