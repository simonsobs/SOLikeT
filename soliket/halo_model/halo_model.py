"""
.. module:: soliket.halo_model

:Synopsis: Class to calculate Halo Models for non-linear power spectra.
:Author: Ian Harrison

.. |br| raw:: html

   <br />


Usage
-----

Halo Models for calculating non-linear power spectra for use in large scale structure 
and lensing likelihoods. The abstract HaloModel base class should be built on with 
specific model implementations. HaloModels can be added as theory codes alongside others 
in your run settings. e.g.:

.. code-block:: yaml

  theory:
    camb:
    soliket.halo_model.HaloModel_pyhm:


Implementing your own halo model
--------------------------------

If you want to add your own halo model, you can do so by inheriting from the
``soliket.HaloModel`` theory class and implementing your own custom ``calculate()``
function (have a look at the simple pyhalomodel model for ideas).
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pyhalomodel as halo
from cobaya.theory import Provider, Theory
# from cobaya.theories.cosmo.boltzmannbase import PowerSpectrumInterpolator
from scipy.interpolate import RectBivariateSpline


class HaloModel(Theory):
    """Abstract parent class for implementing Halo Models."""

    kmax: Union[int, float]
    z: Union[float, List[float], np.ndarray]
    extra_args: Optional[dict]

    _enforce_types: bool = True

    _logz = np.linspace(-3, np.log10(1100), 150)
    _default_z_sampling = 10**_logz
    _default_z_sampling[0] = 0
    provider: Provider

    def initialize(self):
        self._var_pairs = set()
        self._required_results = {}

    # def must_provide(self, **requirements):
    #     options = requirements.get("halo_model") or {}

    def _get_Pk_mm_lin(self) -> np.ndarray:
        for pair in self._var_pairs:
            self.k, self.z, Pk_mm = \
                self.provider.get_Pk_grid(var_pair=pair, nonlinear=False)

        return Pk_mm

    def get_Pk_mm_grid(self) -> np.ndarray:
        return self.current_state["Pk_mm_grid"]

    def get_Pk_gg_grid(self) -> np.ndarray:
        return self.current_state["Pk_gg_grid"]

    def get_Pk_gm_grid(self) -> np.ndarray:
        return self.current_state["Pk_gm_grid"]


class HaloModel_pyhm(HaloModel):
    """Halo Model wrapping the simple pyhalomodel code of Asgari, Mead & Heymans (2023)

    We include this simple halo model for the non-linear matter-matter power spectrum with 
    NFW profiles. This is calculated via the `pyhalomodel
    <https://github.com/alexander-mead/pyhalomodel>`_ code.
    """

    hmf_name: str
    hmf_Dv: float
    Mmin: float
    Mmax: float
    nM: int

    def initialize(self):
        super().initialize()
        self.Ms = np.logspace(np.log10(self.Mmin), np.log10(self.Mmax), self.nM)

    def get_requirements(self) -> Dict[str, Any]:
        return {"omegam": None}

    def must_provide(self, **requirements) -> dict:
        options: dict = requirements.get("halo_model") or {}
        self._var_pairs.update(
            set((x, y) for x, y in
                options.get("vars_pairs", [("delta_tot", "delta_tot")])))

        self.kmax = max(self.kmax, options.get("kmax", self.kmax))
        self.z = np.unique(np.concatenate(
                            (np.atleast_1d(options.get("z", self._default_z_sampling)),
                            np.atleast_1d(self.z))))

        needs = {}

        needs["Pk_grid"] = {
                "vars_pairs": self._var_pairs,
                "nonlinear": (False, False),
                "z": self.z,
                "k_max": self.kmax
            }

        needs["sigma_R"] = {"vars_pairs": self._var_pairs,
                           "z": self.z,
                           "k_max": self.kmax,
                           "R": np.linspace(0.14, 66, 256) # list of radii required
                           }

        return needs

    def calculate(self, state: dict, want_derived: bool = True,
                  **params_values_dict):

        Pk_mm_lin: np.ndarray = self._get_Pk_mm_lin()

        # now wish to interpolate sigma_R to these Rs
        zinterp, rinterp, sigmaRinterp = self.provider.get_sigma_R()
        # sigmaRs = PowerSpectrumInterpolator(zinterp, rinterp, sigma_R)
        sigmaRs = RectBivariateSpline(zinterp, rinterp, sigmaRinterp)

        output_Pk_hm_mm = np.empty([len(self.z), len(self.k)])

        # for sure we could avoid the for loop with some thought
        for iz, zeval in enumerate(self.z):
            hmod = halo.model(zeval, self.provider.get_param('omegam'),
                              name=self.hmf_name, Dv=self.hmf_Dv)

            Rs = hmod.Lagrangian_radius(self.Ms)
            rvs = hmod.virial_radius(self.Ms)

            cs = 7.85 * (self.Ms / 2e12)**-0.081 * (1. + zeval)**-0.71
            Uk = self._win_NFW(self.k, rvs, cs)
            matter_profile = halo.profile.Fourier(self.k, self.Ms, Uk,
                                                  amplitude=self.Ms,
                                                  normalisation=hmod.rhom,
                                                  mass_tracer=True)

            Pk_2h, Pk_1h, Pk_hm = hmod.power_spectrum(self.k, Pk_mm_lin[iz],
                                                      self.Ms, sigmaRs(zeval, Rs)[0],
                                                      {'m': matter_profile},
                                                      verbose=False)

            output_Pk_hm_mm[iz] = Pk_hm['m-m']

        state['Pk_mm_grid'] = output_Pk_hm_mm
        # state['Pk_gm_grid'] = Pk_hm['g-m']
        # state['Pk_gg_grid'] = Pk_hm['g-g']

    def _win_NFW(self, k: np.ndarray, rv: np.ndarray, c: np.ndarray) -> np.ndarray:
        from scipy.special import sici
        rs = rv / c
        kv = np.outer(k, rv)
        ks = np.outer(k, rs)
        Sisv, Cisv = sici(ks + kv)
        Sis, Cis = sici(ks)
        f1 = np.cos(ks) * (Cisv - Cis)
        f2 = np.sin(ks) * (Sisv - Sis)
        f3 = np.sin(kv) / (ks + kv)
        f4 = np.log(1. + c) - c / (1. + c)
        Wk = (f1 + f2 - f3) / f4
        return Wk
