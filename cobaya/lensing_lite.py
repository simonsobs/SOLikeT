import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import binned_statistic as binnedstat

def binner(ls, cls, bin_edges):
    x = ls.copy()
    y = cls.copy()
    numbins = len(bin_edges)-1
    cents = (bin_edges[:-1]+bin_edges[1:])/2.
    bin_edges_min = bin_edges.min()
    bin_edges_max = bin_edges.max()
    y[x<bin_edges_min] = 0
    y[x>bin_edges_max] = 0
    bin_means = binnedstat(x,y,bins=bin_edges,statistic=np.nanmean)[0]
    return cents, bin_means

datafile = 'simulated_clkk_SO_Apr17_mv_nlkk_deproj0_SENS1_fsky_16000_iterOn_20191109/simulated_clkk_SO_Apr17_mv_nlkk_deproj0_SENS1_fsky_16000_iterOn_20191109_sim_00_bandpowers.txt'
covfile = 'simulated_clkk_SO_Apr17_mv_nlkk_deproj0_SENS1_fsky_16000_iterOn_20191109/simulated_clkk_SO_Apr17_mv_nlkk_deproj0_SENS1_fsky_16000_iterOn_20191109_binned_covmat.txt'
lmax = 2400

class LensingLiteLikelihood(object):

    def __init__(self, bandpower_file=datafile, covfile=covfile):
        lefts, rights, bandpowers = np.loadtxt(bandpower_file, unpack=True)
        self.bandpowers = bandpowers
        self.bin_edges = np.append(lefts, [rights[-1]])
        self.cov = np.loadtxt(covfile)
        self.invcov = np.linalg.inv(self.cov)

        self.ell = (lefts + rights)/2

        self.theory_kk = None

    def __call__(self, _theory={'Cl': {'pp': lmax}}):
        cl = _theory.get_Cl(ell_factor=True)
        _, theory_kk = binner(cl['ell'], cl['pp'], self.bin_edges)

        # (y - mu).T x invcov (y - mu)
        resid = self.bandpowers - theory_kk 
#         return -0.5 * np.dot(np.dot(resid.T, self.invcov), resid)
        return multivariate_normal.logpdf(self.bandpowers, mean=theory_kk, cov=self.cov)        
    
