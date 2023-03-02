"""
.. module:: soliket.bias

:Synopsis: Class to calculate bias models for haloes and galaxies as cobaya
Theory classes.
"""

import pdb
import numpy as np
from typing import Sequence, Optional, Union
from cobaya.theory import Theory
from cobaya.typing import InfoDict
from cobaya.log import LoggedError


class Bias(Theory):
    """Parent class for bias models."""

    _logz = np.linspace(-3, np.log10(1100), 150)
    _default_z_sampling = 10**_logz
    _default_z_sampling[0] = 0

    def initialize(self):

        self._var_pairs = set()

    def get_requirements(self):
        return {}

    def must_provide(self, **requirements):

        options = requirements.get("linear_bias") or {}

        self.kmax = max(self.kmax, options.get("kmax", self.kmax))
        self.z = np.unique(np.concatenate(
                            (np.atleast_1d(options.get("z", self._default_z_sampling)),
                            np.atleast_1d(self.z))))

        # Dictionary of the things needed from CAMB/CLASS
        needs = {}

        self.nonlinear = self.nonlinear or options.get("nonlinear", False)
        self._var_pairs.update(
            set((x, y) for x, y in
                options.get("vars_pairs", [("delta_tot", "delta_tot")])))

        needs["Pk_grid"] = {
                "vars_pairs": self._var_pairs or [("delta_tot", "delta_tot")],
                "nonlinear": (True, False) if self.nonlinear else False,
                "z": self.z,
                "k_max": self.kmax
            }

        assert len(self._var_pairs) < 2, "Bias doesn't support other Pk yet"
        return needs

    def _get_Pk_mm(self):
        for pair in self._var_pairs:
            self.k, self.z, Pk_mm = \
                self.provider.get_Pk_grid(var_pair=pair, nonlinear=self.nonlinear)

        return Pk_mm

    def get_Pk_gg_grid(self) -> dict:
        return self._current_state["Pk_gg_grid"]

    def get_Pk_gm_grid(self) -> dict:
        return self._current_state["Pk_gm_grid"]


class Linear_bias(Bias):
    """Linear bias model."""

    def calculate(self, state: dict, want_derived: bool = True,
                  **params_values_dict) -> Optional[bool]:

        Pk_mm = self._get_Pk_mm()

        state["Pk_gg_grid"] = params_values_dict["b_lin"]**2. * Pk_mm
        state["Pk_gm_grid"] = params_values_dict["b_lin"] * Pk_mm
