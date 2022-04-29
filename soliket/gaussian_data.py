import numpy as np

try:
    import holoviews as hv
except ImportError:
    pass
from scipy.linalg import cholesky, LinAlgError


def multivariate_normal_logpdf(theory, data, cov, inv_cov, log_det):
    const = np.log(2 * np.pi) * (-len(data) / 2) + log_det * (-1 / 2)
    delta = data - theory
    #print(const,delta,np.dot(delta, inv_cov.dot(delta)))
    return -0.5 * np.dot(delta, inv_cov.dot(delta)) + const


class GaussianData:
    """Named multivariate gaussian data
    """

    def __init__(self, name, x, y, cov):

        self.name = str(name)

        if not (len(x) == len(y) and cov.shape == (len(x), len(x))):
            raise ValueError(f"Incompatible shapes! x={x.shape}, y={y.shape}, \
                               cov={cov.shape}")

        self.x = x
        self.y = y
        self.cov = cov
        try:
            self.cholesky = cholesky(cov)
        except LinAlgError:
            raise ValueError("Covariance is not SPD!")
        self.inv_cov = np.linalg.inv(self.cov)
        self.log_det = np.linalg.slogdet(self.cov)[1]

    def __len__(self):
        return len(self.x)

    def loglike(self, theory):
        return multivariate_normal_logpdf(theory, self.y, self.cov, self.inv_cov,
                                          self.log_det)


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
                            f"Cross-covariance (for {d1.name} x {d2.name}) \
                              has wrong shape: {cov.shape}!"
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

    def loglike(self, theory):
        return self.data.loglike(theory)

    @property
    def name(self):
        return self.data.name

    @property
    def cov(self):
        return self.data.cov

    @property
    def inv_cov(self):
        return self.data.inv_cov

    @property
    def log_det(self):
        return self.data.log_det

    @property
    def labels(self):
        return [x for y in [[name] * len(d) for
                name, d in zip(self.names, self.data_list)] for x in y]

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

    def plot_cov(self, **kwargs):
        data = [
            (f"{li}: {self.data.x[i]}", f"{lj}: {self.data.x[j]}", self.cov[i, j])
            for i, li in zip(range(len(self.data)), self.labels)
            for j, lj in zip(range(len(self.data)), self.labels)
        ]

        return hv.HeatMap(data).opts(tools=["hover"], width=800, height=800,
                                     invert_yaxis=True, xrotation=90)
