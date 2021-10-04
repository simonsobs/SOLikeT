import numpy as np

from . import utils
from .gaussian import GaussianLikelihood


class PSLikelihood(GaussianLikelihood):
    name: str = "TT"
    kind: str = "tt"
    lmax: int = 6000

    def get_requirements(self):
        return {"Cl": {self.kind: self.lmax}}

    def _get_Cl(self):
        return self.theory.get_Cl(ell_factor=True)

    def _get_theory(self, **params_values):
        cl_theory = self._get_Cl()
        return cl_theory[self.kind][:self.lmax]


class BinnedPSLikelihood(PSLikelihood):
    binning_matrix_path: str = ""

    def initialize(self):
        self.binning_matrix = self._get_binning_matrix()
        self.bin_centers = \
                        self.binning_matrix.dot(np.arange(self.binning_matrix.shape[1]))
        super().initialize()

    @classmethod
    def binner(cls, x, y, bin_edges):
        return utils.binner(x, y, bin_edges)

    def _get_binning_matrix(self):
        return np.loadtxt(self.binning_matrix_path)

    def _get_data(self):
        return self.bin_centers, np.loadtxt(self.datapath)

    def _get_theory(self, **params_values):
        cl_theory = self._get_Cl()
        return self.binning_matrix.dot(cl_theory[self.kind][:self.lmax])
