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
from cobaya.log import LoggedError

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
    k_max: int = 1.
    z: float = np.linspace(0.0, 2, 128)

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
        self.cp_pk_nn = cp.cosmopower_NN(
            restore=True,
            restore_filename=os.path.join(self.soliket_data_path, self.network_path_pk, self.pk_lin_nn_filename),
        )

        if "lmax" not in self.extra_args:
            self.extra_args["lmax"] = None

        self.log.info(f"Loaded CosmoPower from directory {self.network_path}")

    def calculate(self, state, want_derived=True, **params):
        network_params = {}

        for par in self.renames:
            if par in params:
                network_params[par] = [params[par]]
            else:
                for r in self.renames[par]:
                    if r in params:
                        network_params[par] = [params[r]]
                        break

        state["tt"] = self.cp_tt_nn.ten_to_predictions_np(network_params)[0, :]
        state["te"] = self.cp_te_nn.predictions_np(network_params)[0, :]
        state["ee"] = self.cp_ee_nn.ten_to_predictions_np(network_params)[0, :]
        state["ell"] = self.cp_tt_nn.modes

        # if "Pk_grid" in self.requested() or "Pk_interpolator" in self.requested():
        
        network_params_z = {}

        for k in network_params.keys():
            network_params_z[k] = np.repeat(network_params[k], len(self.z))
        network_params_z['z'] = self.z

        var_pair=("delta_tot", "delta_tot")
        nonlinear=True

        state[("Pk_grid", bool(nonlinear)) + tuple(sorted(var_pair))] = self.cp_pk_nn.modes, self.z, self.cp_pk_nn.predictions_np(network_params_z)

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

    # def must_provide(self, **requirements):

    #     super().must_provide(**requirements)

    #     for k, v in self._must_provide.items():
    #          if isinstance(k, tuple) and k[0] == "Pk_grid":
    #             v = deepcopy(v)
    #             self.add_P_k_max(v.pop("k_max"), units="1/Mpc")
    #             # NB: Actually, only the max z is used, and the actual sampling in z
    #             # for computing P(k,z) is controlled by `perturb_sampling_stepsize`
    #             # (default: 0.1). But let's leave it like this in case this changes
    #             # in the future.
    #             self.add_z_for_matter_power(v.pop("z"))
    #             if v["nonlinear"]:
    #                 if "non_linear" not in self.extra_args:
    #                     # this is redundant with initialisation, but just in case
    #                     self.extra_args["non_linear"] = non_linear_default_code
    #                 elif self.extra_args["non_linear"] == non_linear_null_value:
    #                     raise LoggedError(
    #                         self.log, ("Non-linear Pk requested, but `non_linear: "
    #                                    f"{non_linear_null_value}` imposed in "
    #                                    "`extra_args`"))
    #             pair = k[2:]
    #             if pair == ("delta_tot", "delta_tot"):
    #                 self.collectors[k] = Collector(
    #                     method="get_pk_and_k_and_z",
    #                     kwargs=v,
    #                     post=(lambda P, kk, z: (kk, z, np.array(P).T)))
    #             else:
    #                 raise LoggedError(self.log, "NotImplemented in cosmopower: %r", pair)

    #     return needs


    # def get_Pk_grid(self, params, var_pair=("delta_tot", "delta_tot"), nonlinear=False,
    #                         extrap_kmin=None, extrap_kmax=None):
        
    #     return self.current_state['k'], self.current_state['pk_z'], self.current_state['pk']


    def get_can_support_params(self):
        return ["omega_b", "omega_cdm", "h", "logA", "ns", "tau_reio"]

    def get_can_provide(self):
        return ['Cl', 'Pk_grid', 'Pk_interpolator']
