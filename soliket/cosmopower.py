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
    soliket_data_path: str = "soliket/data/CosmoPower"
    network_path: str = "CP_paper/CMB"
    network_path_pk: str = "CP_paper/PK"
    cmb_tt_nn_filename: str = "cmb_TT_NN"
    cmb_te_pcaplusnn_filename: str = "cmb_TE_PCAplusNN"
    cmb_ee_nn_filename: str = "cmb_EE_NN"
    pk_lin_nn_filename: str = "PKLIN_NN"

    extra_args: InfoDict = {}

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

        self.cp_tt_nn = cp.cosmopower_NN(
            restore=True,
            restore_filename=os.path.join(base_path, self.cmb_tt_nn_filename),
        )
        self.cp_te_nn = cp.cosmopower_PCAplusNN(
            restore=True,
            restore_filename=os.path.join(base_path, self.cmb_te_pcaplusnn_filename),
        )
        self.cp_ee_nn = cp.cosmopower_NN(
            restore=True,
            restore_filename=os.path.join(base_path, self.cmb_ee_nn_filename),
        )

        if "lmax" not in self.extra_args:
            self.extra_args["lmax"] = None

        self.log.info(f"Loaded CosmoPower from directory {self.network_path}")

    def calculate(self, state, want_derived=True, **params):
        cmb_params = {}

        for par in self.renames:
            if par in params:
                cmb_params[par] = [params[par]]
            else:
                for r in self.renames[par]:
                    if r in params:
                        cmb_params[par] = [params[r]]
                        break

        state["tt"] = self.cp_tt_nn.ten_to_predictions_np(cmb_params)[0, :]
        state["te"] = self.cp_te_nn.predictions_np(cmb_params)[0, :]
        state["ee"] = self.cp_ee_nn.ten_to_predictions_np(cmb_params)[0, :]
        state["ell"] = self.cp_tt_nn.modes

    def get_Cl(self, ell_factor=False, units="FIRASmuK2"):
        cls_old = self.current_state.copy()

        lmax = self.extra_args["lmax"] or (cls_old["tt"].shape[0] + 2)

        cls = {"ell": np.arange(lmax).astype(int)}
        ls = cls_old["ell"]

        for k in ["tt", "te", "ee"]:
            # cls[k] = np.zeros(cls["ell"].shape, dtype=float)
            cls[k] = np.empty(cls["ell"].shape, dtype=float)
            cls[k][:] = np.nan

        cmb_fac = self._cmb_unit_factor(units, 2.7255)

        if ell_factor:
            ls_fac = ls * (ls + 1.0) / (2.0 * np.pi)
        else:
            ls_fac = 1.0

        for k in ["tt", "te", "ee"]:
            cls[k][ls] = cls_old[k] * ls_fac * cmb_fac ** 2.0
            if np.any(np.isnan(cls[k])):
                self.log.warning("CosmoPower used outside of trained "\
                                 "{} ell range. Filled in with NaNs.".format(k))

        return cls

    def get_can_support_params(self):
        return ["omega_b", "omega_cdm", "h", "logA", "ns", "tau_reio"]
