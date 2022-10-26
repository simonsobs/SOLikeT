# import unittest
import numpy as np
from sklearn.datasets import make_spd_matrix

from soliket.studentst import StudentstData
from soliket.gaussian import GaussianData


def test_studentst():

    name1 = "A"
    n1 = 10
    x1 = np.arange(n1)
    y1th = x1**2.
    y1 = y1th + np.random.random(n1)
    # nsims1 = 50

    cov1 = make_spd_matrix(n1, random_state=1234)

    data1 = GaussianData(name1, x1, y1, cov1)

    nsims_grid = np.logspace(2, 8, 32)

    dL_st_gaussian = []
    dL_hartlap_gaussian = []

    for nsims1 in nsims_grid:

        data1_simcov = GaussianData(name1 + 'simcov',
                                    x1, y1, cov1, ncovsims=nsims1)
        data1_studentst = StudentstData(name1 + 'studentst',
                                        x1, y1, cov1, ncovsims=nsims1)

        # hartlap_factor = (nsims1 - n1 - 2) / (nsims1 - 1)

        # check studetst converging to Hartlap-correct Gaussian
        # dL_st_hartlap = np.abs(data1_studentst.loglike(y1th) -
        #                        data1_simcov.loglike(y1th))
        dL_st_gaussian.append(np.abs(data1_studentst.loglike(y1th) -
                                     data1.loglike(y1th)))
        dL_hartlap_gaussian.append(np.abs(data1_simcov.loglike(y1th) -
                                          data1.loglike(y1th)))

        # check studentst converging to Gaussian faster than Hartlap-corrected Gaussian
        assert dL_st_gaussian[-1] > dL_hartlap_gaussian[-1]

    # from matplotlib import pyplot as plt
    # plt.plot(nsims_grid, dL_hartlap_gaussian, label='Hartlap')
    # plt.plot(nsims_grid, dL_st_gaussian, label='Students-t')
    # plt.legend()
    # plt.xscale('log')
    # # plt.yscale('log')
    # plt.savefig('./studentst.png')
