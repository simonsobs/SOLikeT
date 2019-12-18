import os
from pkg_resources import resource_filename

import numpy as np
from scipy.stats import multivariate_normal

from cobaya.likelihood import Likelihood

from .utils import binner
from .ps import BinnedPSLikelihood


class LensingLikelihood(BinnedPSLikelihood):
    class_options = {"kind": "pp"}


class SimulatedLensingLikelihood(LensingLikelihood):
    class_options = {
        "dataroot": resource_filename(
            "solike", "data/simulated_clkk_SO_Apr17_mv_nlkk_deproj0_SENS1_fsky_16000_iterOn_20191109"
        ),
        "cl_file": "simulated_clkk_SO_Apr17_mv_nlkk_deproj0_SENS1_fsky_16000_iterOn_20191109_sim_{:02d}_bandpowers.txt",
        "cov_file": "simulated_clkk_SO_Apr17_mv_nlkk_deproj0_SENS1_fsky_16000_iterOn_20191109_binned_covmat.txt",
        "sim_number": 0,
    }

    def initialize(self):
        self.datapath = os.path.join(self.dataroot, self.cl_file.format(self.sim_number))
        self.covpath = os.path.join(self.dataroot, self.cov_file)
        super().initialize()
