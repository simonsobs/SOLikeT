import numpy as np
import scipy.stats
import cashstatistic as cashstat

def gof_cash(npred, nobs):
    """
    Computes p-value for Poisson statistic based on Kaastra 2017 (https://arxiv.org/abs/1707.09202).
    Parameters
    ----------
    npred: predicted number of clusters in bins
    nobs: observed number of clusters for same binning
    Returns
    -------
    pval: Gaussian p-value for C-stat
    Ce: expectation value for C-stat
    Cv: variance of C-stat
    Cd: observed value of C-stat
    """

    Ce_bin, Cv_bin = cashstat.cash_mod_expectations(npred)
    Ce = np.sum(Ce_bin)
    Cv = np.sum(Cv_bin)

    logterm = np.zeros_like(nobs, dtype=float)
    logterm[nobs > 0] = np.log(nobs[nobs > 0]/npred[nobs > 0])
    logterm[nobs == 0] = 0.

    Cd = 2*np.sum(npred - nobs + nobs*logterm)

    pval = scipy.stats.norm.sf(Cd, Ce, np.sqrt(Cv))

    return pval, Ce, Cv, Cd
