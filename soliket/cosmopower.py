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

    def initialize(self):
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

        if "lmax" not in self.extra_args:
            self.extra_args["lmax"] = None

        self.log.info(f"Loaded CosmoPower from directory {self.network_path}")

    def calculate(self, state, want_derived=True, **params):
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

    def get_Cl(self, ell_factor=False, units="FIRASmuK2"):
        cls_old = self.current_state.copy()

        lmax = self.extra_args["lmax"] or cls_old["ell"].max()

        cls = {"ell": np.arange(lmax+1).astype(int)}
        ls = cls_old["ell"]

        for k in self.networks:
            # cls[k] = np.zeros(cls["ell"].shape, dtype=float)
            cls[k] = np.empty(cls["ell"].shape, dtype=float)
            cls[k][:] = np.nan

        cmb_fac = self._cmb_unit_factor(units, 2.7255)

        if ell_factor:
            ls_fac = ls * (ls + 1.0) / (2.0 * np.pi)
        else:
            ls_fac = np.ones_like(ls)

        for k in self.networks:
            cl_fac = np.ones_like(ls_fac)
            if self.networks[k]["has_ell_factor"]:
                cl_fac = (2.0 * np.pi) / (ls * (ls + 1.0))

            cls[k][ls] = cls_old[k] * cl_fac * ls_fac * cmb_fac ** 2.0
            if np.any(np.isnan(cls[k])):
                self.log.warning("CosmoPower used outside of trained "\
                                 "{} ell range. Filled in with NaNs.".format(k))

        return cls

    def get_can_support_params(self):
        return self.all_parameters
