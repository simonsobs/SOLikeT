import os
try:
    import cosmopower as cp  # noqa F401
except ImportError:
    HAS_COSMOPOWER = False
else:
    HAS_COSMOPOWER = True
import numpy as np

from cobaya.theories.cosmo import BoltzmannBase
from cobaya.typing import InfoDict

"""
  Simple CosmoPower theory wrapper for Cobaya.
  author: Hidde T. Jense
"""


class CosmoPower(BoltzmannBase):
    stop_at_error: bool = False

    soliket_data_path: str = "soliket/data/CosmoPower"
    network_path: str = "CP_paper/CMB"
    network_settings: InfoDict = { }

    extra_args: InfoDict = { }

    renames: dict = {
        "omega_b": ["ombh2", "omegabh2"],
        "omega_cdm": ["omch2", "omegach2"],
        "ln10^{10}A_s": ["logA"],
        "n_s": ["ns"],
        "h": [],
        "tau_reio": ["tau"],
    }

    def initialize(self) -> None:
        super().initialize()

        base_path = os.path.join(self.soliket_data_path, self.network_path)

        self.networks = { }
        self.all_parameters = set([ ])

        for spectype in self.network_settings:
            netdata = { }
            nettype = self.network_settings[spectype]
            netpath = os.path.join( base_path, nettype["filename"] )

            if nettype["type"] == "NN":
                network = cp.cosmopower_NN( restore = True, restore_filename = netpath )
            elif nettype["type"] == "PCAplusNN":
                network = cp.cosmopower_PCAplusNN( restore = True, restore_filename = netpath )
            elif self.stop_at_error:
                raise ValueError(f"Unknown network type {nettype['type']} for network {spectype}.")
            else:
                self.log.warn(f"Unknown network type {nettype['type']} for network {spectype}: skipped!")

            netdata["type"] = nettype["type"]
            netdata["log"] = nettype.get("log", True)
            netdata["network"] = network
            netdata["parameters"] = list(network.parameters)
            netdata["lmax"] = network.modes.max()
            netdata["has_ell_factor"] = nettype.get("has_ell_factor", False)

            self.all_parameters = self.all_parameters | set(network.parameters)

            if not network is None:
                self.networks[ spectype.lower() ] = netdata

        if "ln10^{10}A_s" in self.all_parameters:
            self.all_parameters.remove("ln10^{10}A_s")
            self.all_parameters.add("logA")

        if "lmax" not in self.extra_args:
            self.extra_args["lmax"] = None

        self.log.info(f"Loaded CosmoPower from directory {self.network_path}")
        self.log.info(f"CosmoPower will expect the parameters {self.all_parameters}")

    def calculate(self, state: dict, want_derived: bool=True, **params) -> dict:
        cmb_params = { }

        for par in self.renames:
            if par in params:
                cmb_params[par] = [params[par]]
            else:
                for r in self.renames[par]:
                    if r in params:
                        cmb_params[par] = [params[r]]
                        break

        ells = None

        for spectype in self.networks:
            network = self.networks[spectype]
            used_params = { par : (cmb_params[par] if par in cmb_params else [params[par]]) for par in network["parameters"] }

            if network["log"]:
                data = network["network"].ten_to_predictions_np(used_params)[0, :]
            else:
                data = network["network"].predictions_np(used_params)[0, :]

            state[spectype] = data

            if ells is None:
                ells = network["network"].modes

        state["ell"] = ells.astype(int)

    def get_Cl(self, ell_factor:bool =False, units:str ="FIRASmuK2") -> dict:
        cls_old = self.current_state.copy()

        lmax = self.extra_args["lmax"] or cls_old["ell"].max()

        cls = {"ell": np.arange(lmax+1).astype(int)}
        ls = cls_old["ell"]
        ls_fac = np.ones_like(ls).astype(float)

        for k in self.networks:
            cls[k] = np.tile(np.nan, cls["ell"].shape)

        for k in self.networks:
            prefac = np.ones_like(ls).astype(float)

            if self.networks[k]["has_ell_factor"]:
                prefac /= self.ell_factor(ls, k)
            if ell_factor:
                prefac *= self.ell_factor(ls, k)

            cls[k][ls] = cls_old[k] * prefac * self.cmb_unit_factor(k, units, 2.7255)
            cls[k][:2] = 0.0
            if np.any(np.isnan(cls[k])):
                self.log.warning("CosmoPower used outside of trained "\
                                 "{} ell range. Filled in with NaNs.".format(k))

        return cls

    def ell_factor(self, ls: np.ndarray, spectra: str) -> np.ndarray:
        ellfac = np.ones_like(ls).astype(float)

        if spectra in [ "tt", "te", "tb", "ee", "et", "eb", "bb", "bt", "be" ]:
            ellfac = ls * (ls + 1.0) / (2.0 * np.pi)
        elif spectra in [ "pt", "pe", "pb", "tp", "ep", "bp" ]:
            ellfac = (ls * (ls + 1.0)) ** (3./2.) / (2.0 * np.pi)
        elif spectra in [ "pp" ]:
            ellfac = (ls * (ls + 1.0)) ** 2.0 / (2.0 * np.pi)

        return ellfac

    def cmb_unit_factor(self, spectra: str, units: str="FIRASmuK2", Tcmb:float =2.7255) -> float:
        res = 1.0
        x,y = spectra

        if x == "t" or x == "e" or x == "b":
            res *= self._cmb_unit_factor(units, Tcmb)
        elif x == "p":
            res *= 1./np.sqrt(2.0*np.pi)

        if y == "t" or y == "e" or y == "b":
            res *= self._cmb_unit_factor(units, Tcmb)
        elif y == "p":
            res *= 1./np.sqrt(2.0*np.pi)

        return res

    def get_can_support_parameters(self) -> list:
        return self.all_parameters

    def get_requirements(self) -> list:
        return list(self.all_parameters)
