import os
from pkg_resources import resource_filename

import numpy as np
from scipy.stats import multivariate_normal

from cobaya.likelihood import Likelihood

from .utils import binner


class LensingLiteLikelihood(Likelihood):
    class_options = {
            'datapath': resource_filename('solike', 'data/simulated_clkk_SO_Apr17_mv_nlkk_deproj0_SENS1_fsky_16000_iterOn_20191109/simulated_clkk_SO_Apr17_mv_nlkk_deproj0_SENS1_fsky_16000_iterOn_20191109_sim_00_bandpowers.txt'),
            'covpath': resource_filename('solike', 'data/simulated_clkk_SO_Apr17_mv_nlkk_deproj0_SENS1_fsky_16000_iterOn_20191109/simulated_clkk_SO_Apr17_mv_nlkk_deproj0_SENS1_fsky_16000_iterOn_20191109_binned_covmat.txt'),
        }


    def initialize(self):

        lefts, rights, bandpowers = np.loadtxt(self.datapath, unpack=True)
        self.bandpowers = bandpowers
        self.bin_edges = np.append(lefts, [rights[-1]])
        self.lmax = int(rights[-1])

        self.cov = np.loadtxt(self.covpath)
        self.norm = multivariate_normal(mean=self.bandpowers, cov=self.cov)

    def get_requirements(self):
        return {'Cl': {'pp': self.lmax}}

    def logp(self, **params_values):
        cl_theory = self.theory.get_Cl(ell_factor=True)
        _, theory_kk = binner(cl_theory['ell'], cl_theory['pp'], self.bin_edges)
        return self.norm.logpdf(theory_kk)


class SimulatedLensingLiteLikelihood(LensingLiteLikelihood):
    class_options = {
        'dataroot': resource_filename('solike', 'data/simulated_clkk_SO_Apr17_mv_nlkk_deproj0_SENS1_fsky_16000_iterOn_20191109'),
        'cl_file': 'simulated_clkk_SO_Apr17_mv_nlkk_deproj0_SENS1_fsky_16000_iterOn_20191109_sim_{:02d}_bandpowers.txt',
        'cov_file': 'simulated_clkk_SO_Apr17_mv_nlkk_deproj0_SENS1_fsky_16000_iterOn_20191109_binned_covmat.txt',
        'sim_number': 0,
    }

    def initialize(self):
        self.datapath = os.path.join(self.dataroot, self.cl_file.format(self.sim_number))
        self.covpath = os.path.join(self.dataroot, self.cov_file)
        super().initialize()


class LensingLiteLikelihood2(LensingLiteLikelihood):
    class_options = {'testpar': 5}

