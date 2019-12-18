import numpy as np
from sklearn.datasets import make_spd_matrix
from scipy.stats import multivariate_normal

from solike.gaussian import GaussianData
from solike import GaussianLikelihood, MultiGaussianLikelihood
from solike import PSLikelihood


class ToyLikelihood(PSLikelihood):
    class_options = {
        "name": "toy",
        "n": 10,
        "sigma": 1,
        "off_diag_amp": 1e-3,
        "cov": None,
    }

    def initialize(self):
        x = np.arange(self.n)
        if self.cov is None:
            cov = make_spd_matrix(self.n) * self.off_diag_amp
            cov += np.diag(np.ones(self.n) * self.sigma ** 2)
        else:
            cov = self.cov

        y = np.random.multivariate_normal(np.zeros(self.n), cov)
        self.data = GaussianData(self.name, x, y, cov)

    def _get_theory(self):
        return np.zeros(self.n)


def test_toy():
    n1, n2, n3 = [10, 20, 30]
    full_cov = make_spd_matrix(n1 + n2 + n3, random_state=1234) * 1e-1
    full_cov += np.diag(np.ones((n1 + n2 + n3)))

    cov1 = full_cov[:n1, :n1]
    cov2 = full_cov[n1 : n1 + n2, n1 : n1 + n2]
    cov3 = full_cov[n1 + n2 :, n1 + n2 :]

    name1, name2, name3 = ["A", "B", "C"]

    cross_cov = {
        (name1, name2): full_cov[:n1, n1 : n1 + n2],
        (name1, name3): full_cov[:n1, n1 + n2 :],
        (name2, name3): full_cov[n1 : n1 + n2, n1 + n2 :],
    }

    info1 = {"name": name1, "n": n1, "cov": cov1}
    like1 = ToyLikelihood(info1)

    info2 = {"name": name2, "n": n2, "cov": cov2}
    like2 = ToyLikelihood(info2)

    info3 = {"name": name3, "n": n3, "cov": cov3}
    like3 = ToyLikelihood(info3)

    multilike1 = MultiGaussianLikelihood([like1, like2, like3])
    multilike2 = MultiGaussianLikelihood([like1, like2, like3], cross_cov)

    assert np.isclose(multilike1.logp(), sum([l.logp() for l in [like1, like2, like3]]))
    assert not np.isclose(multilike2.logp(), sum([l.logp() for l in [like1, like2, like3]]))
