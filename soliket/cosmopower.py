"""
.. module:: soliket.cosmopower

:Synopsis: Simple CosmoPower theory wrapper for Cobaya.
:Author: Hidde T. Jense

.. |br| raw:: html

   <br />

.. note::

   **If you use this cosmological code, please cite:**
   |br|
   A. Spurio Mancini et al.
   *CosmoPower: emulating cosmological power spectra for accelerated Bayesian
   inference from next-generation surveys*
   (`arXiv:210603846 <https://arxiv.org/abs/2106.03846>`_)

   And remember to cite any sources for trained networks you use.

Usage
-----

After installing SOLikeT and cosmopower, you can use the ``CosmoPower`` theory codes
by adding the ``soliket.CosmoPower`` code as a block in your parameter files.

Example: CMB emulators
----------------------

You can get the example CMB emulators from the `cosmopower release repository <https://github.com/alessiospuriomancini/cosmopower/tree/main/cosmopower/trained_models/CP_paper>`_.
After downloading these, you should have a directory structure like:

.. code-block:: bash

  /path/to/cosmopower/data
    ├── cmb_TT_NN.pkl
    ├── cmb_TE_PCAplusNN.pkl
    └── cmb_EE_NN.pkl

With these and with ``soliket.CosmoPower`` installed and visible to cobaya, you can add it as a theory block to your run yaml as:

.. code-block:: yaml

  theory:
    soliket.CosmoPower:
      network_path: /path/to/cosmopower/data
      network_settings:
        tt:
          type: NN
          filename: cmb_TT_NN
        te:
          type: PCAplusNN
          filename: cmb_TE_PCAplusNN
          log: False
        ee:
          type: NN
          filename: cmb_EE_NN

Running this with cobaya will use ``soliket.CosmoPower`` as a theory to calculate the CMB Cl's from the emulators.

If you want to add the example PP networks as well, you can do that simply with a block as:

.. code-block:: yaml

  theory:
    soliket.CosmoPower:
      network_path: /path/to/cosmopower/data
      network_settings:
        pp:
          type: PCAplusNN
          filename: cmb_PP_PCAplusNN

SOLikeT will automatically use the correct conversion prefactors :math:`\ell (\ell + 1) / 2 \pi` terms and similar, as well as the CMB temperature.
See the :func:`~soliket.cosmopower.CosmoPower.ell_factor` and :func:`~soliket.cosmopower.CosmoPower.cmb_unit_factor` functions for more information.
"""
import os
try:
    import cosmopower as cp  # noqa F401
except ImportError:
    HAS_COSMOPOWER = False
else:
    HAS_COSMOPOWER = True
import numpy as np

from typing import Dict, Iterable, Tuple

from cobaya.log import LoggedError
from cobaya.theory import Theory
from cobaya.theories.cosmo import BoltzmannBase
from cobaya.typing import InfoDict


class CosmoPower(BoltzmannBase):
    """A CosmoPower Network wrapper for Cobaya."""

    def initialize(self) -> None:
        super().initialize()

        if self.network_settings is None:
            raise LoggedError("No network settings were provided.")

        self.networks = {}
        self.all_parameters = set([])

        for spectype in self.network_settings:
            netdata = {}
            nettype = self.network_settings[spectype]
            netpath = os.path.join(self.network_path, nettype["filename"])

            if nettype["type"] == "NN":
                network = cp.cosmopower_NN(
                    restore=True, restore_filename=netpath)
            elif nettype["type"] == "PCAplusNN":
                network = cp.cosmopower_PCAplusNN(
                    restore=True, restore_filename=netpath)
            elif self.stop_at_error:
                raise ValueError(
                    f"Unknown network type {nettype['type']} for network {spectype}.")
            else:
                self.log.warn(
                    f"Unknown network type {nettype['type']}\
                                                for network {spectype}: skipped!")

            netdata["type"] = nettype["type"]
            netdata["log"] = nettype.get("log", True)
            netdata["network"] = network
            netdata["parameters"] = list(network.parameters)
            netdata["lmax"] = network.modes.max()
            netdata["has_ell_factor"] = nettype.get("has_ell_factor", False)

            self.all_parameters = self.all_parameters | set(network.parameters)

            if network is not None:
                self.networks[spectype.lower()] = netdata

        if "lmax" not in self.extra_args:
            self.extra_args["lmax"] = None

        self.log.info(f"Loaded CosmoPower from directory {self.network_path}")
        self.log.info(
            f"CosmoPower will expect the parameters {self.all_parameters}")

    def calculate(self, state: dict, want_derived: bool = True, **params) -> bool:
        ## sadly, this syntax not valid until python 3.9
        # cmb_params = {
        #     p: [params[p]] for p in params
        # } | {
        #     self.translate_param(p): [params[p]] for p in params
        # }
        cmb_params = {**{
            p: [params[p]] for p in params
        }, **{
            self.translate_param(p): [params[p]] for p in params
        }}

        ells = None

        for spectype in self.networks:
            network = self.networks[spectype]
            used_params = {par: (cmb_params[par] if par in cmb_params else [
                                 params[par]]) for par in network["parameters"]}

            if network["log"]:
                data = network["network"].ten_to_predictions_np(used_params)[
                    0, :]
            else:
                data = network["network"].predictions_np(used_params)[0, :]

            state[spectype] = data

            if ells is None:
                ells = network["network"].modes

        state["ell"] = ells.astype(int)

        return True

    def get_Cl(self, ell_factor: bool = False, units: str = "FIRASmuK2") -> dict:
        cls_old = self.current_state.copy()

        lmax = self.extra_args["lmax"] or cls_old["ell"].max()

        cls = {"ell": np.arange(lmax + 1).astype(int)}
        ls = cls_old["ell"]

        for k in self.networks:
            cls[k] = np.tile(np.nan, cls["ell"].shape)

        for k in self.networks:
            prefac = np.ones_like(ls).astype(float)

            if self.networks[k]["has_ell_factor"]:
                prefac /= self.ell_factor(ls, k)
            if ell_factor:
                prefac *= self.ell_factor(ls, k)

            cls[k][ls] = cls_old[k] * prefac * \
                self.cmb_unit_factor(k, units, 2.7255)
            cls[k][:2] = 0.0
            if np.any(np.isnan(cls[k])):
                self.log.warning("CosmoPower used outside of trained "
                                 "{} ell range. Filled in with NaNs.".format(k))

        return cls

    def ell_factor(self, ls: np.ndarray, spectra: str) -> np.ndarray:
        """
        Calculate the ell factor for a specific spectrum.
        These prefactors are used to convert from Cell to Dell and vice-versa.

        See also:
        cobaya.BoltzmannBase.get_Cl
        `camb.CAMBresults.get_cmb_power_spectra <https://camb.readthedocs.io/en/latest/results.html#camb.results.CAMBdata.get_cmb_power_spectra>`_ # noqa E501

        Example:
        ell_factor(l, "tt") -> l(l+1)/(2 pi).
        ell_factor(l, "pp") -> l^2(l+1)^2/(2 pi).

        :param ls: the range of ells.
        :param spectra: a two-character string with each character being one of [tebp].

        :return: an array filled with ell factors for the given spectrum.
        """
        ellfac = np.ones_like(ls).astype(float)

        if spectra in ["tt", "te", "tb", "ee", "et", "eb", "bb", "bt", "be"]:
            ellfac = ls * (ls + 1.0) / (2.0 * np.pi)
        elif spectra in ["pt", "pe", "pb", "tp", "ep", "bp"]:
            ellfac = (ls * (ls + 1.0)) ** (3. / 2.) / (2.0 * np.pi)
        elif spectra in ["pp"]:
            ellfac = (ls * (ls + 1.0)) ** 2.0 / (2.0 * np.pi)

        return ellfac

    def cmb_unit_factor(self, spectra: str,
                        units: str = "FIRASmuK2",
                        Tcmb: float = 2.7255) -> float:
        """
        Calculate the CMB prefactor for going from dimensionless power spectra to
        CMB units.

        :param spectra: a length 2 string specifying the spectrum for which to
                        calculate the units.
        :param units: a string specifying which units to use.
        :param Tcmb: the used CMB temperature [units of K].
        :return: The CMB unit conversion factor.
        """
        res = 1.0
        x, y = spectra.lower()

        if x == "t" or x == "e" or x == "b":
            res *= self._cmb_unit_factor(units, Tcmb)
        elif x == "p":
            res *= 1. / np.sqrt(2.0 * np.pi)

        if y == "t" or y == "e" or y == "b":
            res *= self._cmb_unit_factor(units, Tcmb)
        elif y == "p":
            res *= 1. / np.sqrt(2.0 * np.pi)

        return res

    def get_can_support_parameters(self) -> Iterable[str]:
        return self.all_parameters

    def get_requirements(self) -> Iterable[Tuple[str, str]]:
        requirements = []
        for k in self.all_parameters:
            if k in self.renames.values():
                for v in self.renames:
                    if self.renames[v] == k:
                        requirements.append((v, None))
                        break
            else:
                requirements.append((k, None))

        return requirements


class CosmoPowerDerived(Theory):
    """A theory class that can calculate derived parameters from CosmoPower networks."""

    def initialize(self) -> None:
        super().initialize()

        if self.network_settings is None:
            raise LoggedError("No network settings were provided.")

        netpath = os.path.join(self.network_path, self.network_settings["filename"])

        if self.network_settings["type"] == "NN":
            self.network = cp.cosmopower_NN(
                restore=True, restore_filename=netpath)
        elif self.network_settings["type"] == "PCAplusNN":
            self.network = cp.cosmopower_PCAplusNN(
                restore=True, restore_filename=netpath)
        else:
            raise LoggedError(
                f"Unknown network type {self.network_settings['type']}.")

        self.input_parameters = set(self.network.parameters)

        self.log_data = self.network_settings.get("log", False)

        self.log.info(
            f"Loaded CosmoPowerDerived from directory {self.network_path}")
        self.log.info(
            f"CosmoPowerDerived will expect the parameters {self.input_parameters}")
        self.log.info(
            f"CosmoPowerDerived can provide the following parameters: \
                                                            {self.get_can_provide()}.")

    def translate_param(self, p):
        return self.renames.get(p, p)

    def calculate(self, state: dict, want_derived: bool = True, **params) -> bool:
        ## sadly, this syntax not valid until python 3.9
        # input_params = {
        #     p: [params[p]] for p in params
        # } | {
        #     self.translate_param(p): [params[p]] for p in params
        # }
        input_params = {**{
            p: [params[p]] for p in params
        }, **{
            self.translate_param(p): [params[p]] for p in params
        }}

        if self.log_data:
            data = self.network.ten_to_predictions_np(input_params)[0, :]
        else:
            data = self.network.predictions_np(input_params)[0, :]

        for k, v in zip(self.derived_parameters, data):
            if len(k) == 0 or k == "_":
                continue

            state["derived"][k] = v

        return True

    def get_param(self, p) -> float:
        return self.current_state["derived"][self.translate_param(p)]

    def get_can_support_parameters(self) -> Iterable[str]:
        return self.input_parameters

    def get_requirements(self) -> Iterable[Tuple[str, str]]:
        requirements = []
        for k in self.input_parameters:
            if k in self.renames.values():
                for v in self.renames:
                    if self.renames[v] == k:
                        requirements.append((v, None))
                        break
            else:
                requirements.append((k, None))

        return requirements

    def get_can_provide(self) -> Iterable[str]:
        return set([par for par in self.derived_parameters
                                    if (len(par) > 0 and not par == "_")])
