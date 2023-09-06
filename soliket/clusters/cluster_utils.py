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

    Note that cashstatistic doesn't handle the case of mu=0,
    I had to edit the function cash_mod_expectations
    in cashstatistic.py slightly:

      mu = np.asarray(mu_in)
      lnmu = np.empty(mu.shape)
      lnmu[mu > 0] = np.log(mu[mu > 0])
      lnmu[mu == 0] = 0.
      mi = np.empty(mu.shape)
      mi[mu > 0] = 1.0/mu[mu > 0]
      mi[mu == 0] = 1.
      C_e = np.empty(mu.shape)
      C_v = np.empty(mu.shape)
      C_e[mu == 0.0] = 0.
      C_v[mu == 0.0] = 0.

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
