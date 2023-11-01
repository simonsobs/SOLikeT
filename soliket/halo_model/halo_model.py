import numpy as np
from typing import Optional
from cobaya.theory import Theory
from cobaya.typing import InfoDict

class HaloModel(Theory):
    """Parent class for bias models."""

    def initialize(self):

    def get_requirements(self):

        return

    def must_provide(self):

        return needs

    def _get_Pk_mm_lin(self):
        for pair in self._var_pairs:
            self.k, self.z, Pk_mm = \
                self.provider.get_Pk_grid(var_pair=pair, nonlinear=False)

        return Pk_mm

    def get_Pk_mm_grid(self):

        return self._current_state("Pk_mm_grid")

    def get_Pk_gg_grid(self):

        return self._current_state("Pk_gg_grid")

    def get_Pk_gm_grid(self):

        return self._current_state("Pk_gm_grid")

    def Pk_1h_mm(self):

        return

    def Pk_2h_mm(self):

        Pk_mm_lin = self._get_Pk_mm_lin()

        return Pk_mm_lin * integral

    
    def Pk_gg(self):

        return self.Pk_cc_gg() + 2. * self.Pk_cs_gg() + self.Pk_ss_gg()

    def Pk_cc_gg(self):

        return integral

    def Pk_cs_gg(self):

        return integral

    def Pk_ss_gg(self):

        return integral



