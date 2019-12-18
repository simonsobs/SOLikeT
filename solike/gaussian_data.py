import numpy as np
from scipy.linalg import cholesky, LinAlgError
from scipy.stats import multivariate_normal


class GaussianData(object):
    """Named multivariate gaussian data

    For CMB PS data, x will typically be l, and y will be power spectrum.
    """

    def __init__(self, name, x, y, cov):

        self.name = str(name)

        if not (len(x) == len(y) and cov.shape == (len(x), len(x))):
            raise ValueError(f"Incompatible shapes! x={x.shape}, y={y.shape}, cov={cov.shape}")

        self.x = x
        self.y = y
        self.cov = cov
        try:
            self.cholesky = cholesky(cov)
        except LinAlgError:
            raise ValueError("Covariance is not SPD!")
        self.norm = multivariate_normal(self.y, cov=self.cov)

    def __len__(self):
        return len(self.x)


class MultiGaussianData(GaussianData):
    """

    Parameters
    ----------
    data_list : list
        List of Data objects

    cross_covs : dictionary
        Cross-covariances, keyed by (name1, name2) tuples.
    """

    def __init__(self, data_list, cross_covs=None):

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
                            f"Cross-covariance (for {d1.name} x {d2.name}) has wrong shape: {cov.shape}!"
                        )
                elif rev_key in cross_covs:
                    cross_covs[key] = cross_covs[rev_key].T
                else:
                    cross_covs[key] = np.zeros((len(d1), len(d2)))

        self.data_list = data_list
        self.lengths = [len(d) for d in data_list]
        self.names = [d.name for d in data_list]
        self.cross_covs = cross_covs

        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._assemble_data()
        return self._data

    @property
    def norm(self):
        return self.data.norm

    def _index_range(self, name):
        if name not in self.names:
            raise ValueError(f"{name} not in {self.names}!")

        i0 = 0
        for n, length in zip(self.names, self.lengths):
            if n == name:
                i1 = i0 + length
                break
            i0 += length
        return i0, i1

    def _slice(self, *names):
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
