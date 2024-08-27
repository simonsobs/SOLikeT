from typing import Dict, Tuple
from cobaya.theory import Provider
import numpy as np

from soliket import utils
from soliket.gaussian import GaussianLikelihood


class PSLikelihood(GaussianLikelihood):
    name: str = "TT"
    kind: str = "tt"
    lmax: int = 6000
    provider: Provider

    def get_requirements(self) -> Dict[str, Dict[str, int]]:
        return {"Cl": {self.kind: self.lmax}}

    def _get_Cl(self) -> Dict[str, np.ndarray]:
        return self.provider.get_Cl(ell_factor=True)

    def _get_theory(self, **params_values: dict) -> np.ndarray:
        cl_theory = self._get_Cl()
        return cl_theory[self.kind][:self.lmax]


class BinnedPSLikelihood(PSLikelihood):
    binning_matrix_path: str = ""

    def initialize(self) -> None:
        self.binning_matrix = self._get_binning_matrix()
        self.bin_centers = self.binning_matrix.dot(
            np.arange(self.binning_matrix.shape[1])
        )
        super().initialize()

    @classmethod
    def binner(
        cls, x: np.ndarray, y: np.ndarray, bin_edges: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return utils.binner(x, y, bin_edges)

    def _get_binning_matrix(self) -> np.ndarray:
        return np.loadtxt(self.binning_matrix_path)

    def _get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.bin_centers, np.loadtxt(self.datapath)

    def _get_theory(self, **params_values: dict) -> np.ndarray:
        cl_theory: Dict[str, np.ndarray] = self._get_Cl()
        return self.binning_matrix.dot(cl_theory[self.kind][:self.lmax])
