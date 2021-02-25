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
        return cl_theory[self.kind]


class BinnedPSLikelihood(PSLikelihood):

    binning_matrix_file: str = ""

    def initialize(self):
        super().initialize()
        self.binning_matrix = self._get_binning_matrix()

    @classmethod
    def binner(cls, x, y, bin_edges):
        return utils.binner(x, y, bin_edges)

    def _get_binning_matrix(self):
        return np.loadtxt(self.binning_matrix_file)

    def _get_data(self):
        lefts, rights, bandpowers = np.loadtxt(self.datapath, unpack=True)

        bin_centers = (lefts + rights) / 2
        self.bin_edges = np.append(lefts, [rights[-1]])

        return bin_centers, bandpowers

    def _get_theory(self, **params_values):
        cl_theory = self._get_Cl()
        _, theory = self.binner(cl_theory["ell"], cl_theory[self.kind], self.bin_edges)
        return theory
