import numpy as np

from .gaussian_data import GaussianData


def multivariate_studentst_logpdf(theory, data, cov, inv_cov, log_det, ncovsims):
    const = np.log(2 * np.pi) * (-len(data) / 2) + log_det * (-1 / 2)
    delta = data - theory
    #print(const,delta,np.dot(delta, inv_cov.dot(delta)))
    chi2 = np.dot(delta, inv_cov.dot(delta))
    return -0.5 * ncovsims * np.log(1. - chi2 / (1. - ncovsims)) + const


class StudentstData(GaussianData):
    """Named multivariate gaussian data
    """

    def loglike(self, theory):
        return multivariate_studentst_logpdf(theory, self.y, self.cov, self.inv_cov,
                                             self.log_det, self.ncovsims)
