import numpy as np

from .utils import binner
from .gaussian import GaussianLikelihood


class PSLikelihood(GaussianLikelihood):

    class_options = {"name": "TT", "kind": "tt", "lmax": 6000}

    def get_requirements(self):
        return {"Cl": {self.kind: self.lmax}}

    def _get_Cl(self):
        return self.theory.get_Cl(ell_factor=True)

    def _get_theory(self, **params_values):
        cl_theory = self._get_Cl()
        return cl_theory[self.kind]


class BinnedPSLikelihood(PSLikelihood):
    # def _get_lmax(self):
    #     return int(self.bin_edges[-1])

    def _get_data(self):
        lefts, rights, bandpowers = np.loadtxt(self.datapath, unpack=True)

        bin_centers = (lefts + rights) / 2
        self.bin_edges = np.append(lefts, [rights[-1]])

        return bin_centers, bandpowers

    def _get_theory(self, **params_values):
        cl_theory = self._get_Cl()
        _, theory = binner(cl_theory["ell"], cl_theory[self.kind], self.bin_edges)
        return theory
