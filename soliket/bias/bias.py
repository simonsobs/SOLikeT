"""
.. module:: soliket.bias

:Synopsis: Class to calculate bias models for haloes and galaxies as cobaya
Theory classes.
:author: Ian Harrison

Usage
-----

To use the Linear Bias model, simply add it as a theory code alongside camb in
your run settings, e.g.:

.. code-block:: yaml

  theory:
    camb:
    soliket.bias.linear_bias:


Implementing your own bias model
--------------------------------

If you want to add your own bias model, you can do so by inheriting from the
``soliket.Bias`` theory class and implementing your own custom ``calculate()``
function (have a look at the linear bias model for ideas).
"""

from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from cobaya.theory import Theory


class Bias(Theory):
    """Parent class for bias models."""

    kmax: Union[int, float]
    nonlinear: bool
    z: Union[float, List[float], np.ndarray]
    extra_args: Optional[dict]
    params: dict

    _logz = np.linspace(-3, np.log10(1100), 150)
    _default_z_sampling = 10 ** _logz
    _default_z_sampling[0] = 0

    def initialize(self) -> None:
        self.validate_attributes({k: getattr(self, k) for k in self.get_annotations()})
        self._var_pairs: Set[Tuple[str, str]] = set()

    def get_requirements(self) -> Dict[str, dict]:
        return {}

    def _validate_type(self, expected_type, value):
        if hasattr(expected_type, "__origin__"):
            origin = expected_type.__origin__
            args = expected_type.__args__

            if origin is Union:
                return any(self._validate_type(t, value) for t in args)
            elif origin is Optional:
                return value is None or self._validate_type(args[0], value)
            elif origin is list:
                return all(self._validate_type(args[0], item) for item in value)
            elif origin is dict:
                return all(
                    self._validate_type(args[0], k) and self._validate_type(args[1], v)
                    for k, v in value.items()
                )
            elif origin is tuple:
                return len(args) == len(value) and all(
                    self._validate_type(t, v) for t, v in zip(args, value)
                )
            else:
                return isinstance(value, origin)
        else:
            return isinstance(value, expected_type)

    def _validate_attribute(self, name, value, annotations):
        if name in annotations:
            expected_type = annotations[name]
            if expected_type is float:
                expected_type = Union[int, float]
            if not self._validate_type(expected_type, value):
                msg = f"Attribute '{name}' must be of type \
                        {expected_type}, not {type(value)}"
                raise TypeError(msg)

    def validate_attributes(self, attributes: dict):
        annotations = self.get_annotations()
        for name, value in attributes.items():
            self._validate_attribute(name, value, annotations)

    def must_provide(self, **requirements: dict) -> Dict[str, dict]:
        options = requirements.get("linear_bias") or {}

        self.kmax = max(self.kmax, options.get("kmax", self.kmax))
        self.z = np.unique(np.concatenate(
            (np.atleast_1d(options.get("z", self._default_z_sampling)),
             np.atleast_1d(self.z))))

        # Dictionary of the things needed from CAMB/CLASS
        needs: Dict[str, dict] = {}

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

    def _get_Pk_mm(self) -> np.ndarray:
        self.k, self.z, Pk_mm = \
            self.provider.get_Pk_grid(var_pair=list(self._var_pairs)[0],
                                      nonlinear=self.nonlinear)
        return Pk_mm

    def get_Pk_gg_grid(self) -> dict:
        return self._current_state["Pk_gg_grid"]

    def get_Pk_gm_grid(self) -> dict:
        return self._current_state["Pk_gm_grid"]


class Linear_bias(Bias):
    r"""
    :Synopsis: Linear bias model.

    Has one free parameter, :math:`b_\mathrm{lin}` (``b_lin``).
    """

    params: dict

    def initialize(self) -> None:
        self.validate_attributes({k: getattr(self, k) for k in self.get_annotations()})
        super().initialize()

    def calculate(self, state: dict, want_derived: bool = True,
                  **params_values_dict) -> None:
        Pk_mm = self._get_Pk_mm()

        state["Pk_gg_grid"] = params_values_dict["b_lin"] ** 2. * Pk_mm
        state["Pk_gm_grid"] = params_values_dict["b_lin"] * Pk_mm
