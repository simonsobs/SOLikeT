import unittest
import numpy as np
from sklearn.datasets import make_spd_matrix

from soliket.gaussian import GaussianData, MultiGaussianData, CrossCov


def toy_data():
    name1 = "A"
    n1 = 10
    x1 = np.arange(n1)
    y1 = np.random.random(n1)

    name2 = "B"
    n2 = 20
    x2 = np.arange(n2)
    y2 = np.random.random(n2)

    name3 = "C"
    n3 = 30
    x3 = np.arange(n3)
    y3 = np.random.random(n3)

    # Generate arbitrary covariance matrix, partition into parts
    full_cov = make_spd_matrix(n1 + n2 + n3, random_state=1234)
    cov1 = full_cov[:n1, :n1]
    cov2 = full_cov[n1: n1 + n2, n1: n1 + n2]
    cov3 = full_cov[n1 + n2:, n1 + n2:]

    data1 = GaussianData(name1, x1, y1, cov1)
    data2 = GaussianData(name2, x2, y2, cov2)
    data3 = GaussianData(name3, x3, y3, cov3)

    cross_cov = CrossCov(
        {
            (name1, name2): full_cov[:n1, n1: n1 + n2],
            (name1, name3): full_cov[:n1, n1 + n2:],
            (name2, name3): full_cov[n1: n1 + n2, n1 + n2:],
        }
    )

    return [data1, data2, data3], cross_cov


def test_gaussian():
    datalist, cross_cov = toy_data()

    multi = MultiGaussianData(datalist, cross_cov)

    name1, name2, name3 = [d.name for d in datalist]
    data1, data2, data3 = datalist

    assert (multi.cross_covs[(name1, name2)] == multi.cross_covs[(name2, name1)].T).all()
    assert (multi.cross_covs[(name1, name3)] == multi.cross_covs[(name3, name1)].T).all()
    assert (multi.cross_covs[(name2, name3)] == multi.cross_covs[(name3, name2)].T).all()

    assert (multi.cross_covs[(name1, name1)] == data1.cov).all()
    assert (multi.cross_covs[(name2, name2)] == data2.cov).all()
    assert (multi.cross_covs[(name3, name3)] == data3.cov).all()
