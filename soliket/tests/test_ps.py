import os
from tempfile import gettempdir

import numpy as np
from sklearn.datasets import make_spd_matrix

from soliket import MultiGaussianLikelihood, PSLikelihood
from soliket.gaussian import CrossCov, GaussianData
from soliket.utils import get_likelihood


class ToyLikelihood(PSLikelihood):
    name = "toy"
    n = 10
    sigma = 1
    off_diag_amp = 1e-3
    cov = None
    seed = 1234

    def initialize(self):
        np.random.seed(self.seed)

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
    full_cov += np.diag(np.ones(n1 + n2 + n3))

    cov1 = full_cov[:n1, :n1]
    cov2 = full_cov[n1: n1 + n2, n1: n1 + n2]
    cov3 = full_cov[n1 + n2:, n1 + n2:]

    name1, name2, name3 = ["A", "B", "C"]

    cross_cov = CrossCov(
        {
            (name1, name2): full_cov[:n1, n1: n1 + n2],
            (name1, name3): full_cov[:n1, n1 + n2:],
            (name2, name3): full_cov[n1: n1 + n2, n1 + n2:],
        }
    )
    tempdir = gettempdir()
    cross_cov_path = os.path.join(tempdir, "toy_cross_cov.npz")
    cross_cov.save(cross_cov_path)

    info1 = {"name": name1, "n": n1, "cov": cov1, "seed": 123}
    info2 = {"name": name2, "n": n2, "cov": cov2, "seed": 234}
    info3 = {"name": name3, "n": n3, "cov": cov3, "seed": 345}

    lhood = "soliket.tests.test_ps.ToyLikelihood"
    components = [lhood] * 3
    options = [info1, info2, info3]
    multilike1 = MultiGaussianLikelihood({"components": components, "options": options})
    multilike2 = MultiGaussianLikelihood(
        {"components": components, "options": options, "cross_cov_path": cross_cov_path}
    )

    like1 = get_likelihood(lhood, info1)
    like2 = get_likelihood(lhood, info2)
    like3 = get_likelihood(lhood, info3)

    assert np.isclose(multilike1.logp(), sum([likex.logp() for
                                              likex in [like1, like2, like3]]))
    assert not np.isclose(multilike2.logp(), sum([likex.logp() for
                                                  likex in [like1, like2, like3]]))
