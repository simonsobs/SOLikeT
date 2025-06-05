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

import numpy as np
import pyhalomodel as halo
from cobaya.theory import Theory

# from cobaya.theories.cosmo.boltzmannbase import PowerSpectrumInterpolator
from scipy.interpolate import RectBivariateSpline


class HaloModel(Theory):
    """Abstract parent class for implementing Halo Models."""

    _logz = np.linspace(-3, np.log10(1100), 150)
    _default_z_sampling = 10**_logz
    _default_z_sampling[0] = 0

    def initialize(self):
        self._var_pairs = set()
        self._required_results = {}

    def _get_Pk_mm_lin(self):
        for pair in self._var_pairs:
            self.k, self.z, pk_mm = self.provider.get_Pk_grid(
                var_pair=pair, nonlinear=False
            )
        return pk_mm

    def get_Pk_mm_grid(self):
        return self.current_state["Pk_mm_grid"]

    def get_Pk_gg_grid(self):
        return self.current_state["Pk_gg_grid"]

    def get_Pk_gm_grid(self):
        return self.current_state["Pk_gm_grid"]


class HaloModel_pyhm(HaloModel):
    """Halo Model wrapping the simple pyhalomodel code of Asgari, Mead & Heymans (2023)

    We include this simple halo model for the non-linear matter-matter power spectrum with
    NFW profiles. This is calculated via the `pyhalomodel
    <https://github.com/alexander-mead/pyhalomodel>`_ code.
    """

    def initialize(self):
        super().initialize()
        self.Ms: np.ndarray = np.logspace(
            np.log10(self.Mmin), np.log10(self.Mmax), self.nM
        )

    def get_requirements(self):
        return {"omegam": None}

    def must_provide(self, **requirements):
        options = requirements.get("halo_model") or {}
        self._var_pairs.update(
            {(x, y) for x, y in options.get("vars_pairs", [("delta_tot", "delta_tot")])}
        )

        self.kmax = max(self.kmax, options.get("kmax", self.kmax))
        self.z = np.unique(
            np.concatenate(
                (
                    np.atleast_1d(options.get("z", self._default_z_sampling)),
                    np.atleast_1d(self.z),
                )
            )
        )

        needs = {}

        needs["Pk_grid"] = {
            "vars_pairs": self._var_pairs,
            "nonlinear": (False, False),
            "z": self.z,
            "k_max": self.kmax,
        }

        needs["sigma_R"] = {
            "vars_pairs": self._var_pairs,
            "z": self.z,
            "k_max": self.kmax,
            "R": np.linspace(0.14, 66, 256),  # list of radii required
        }

        return needs

    def calculate(self, state: dict, want_derived: bool = True, **params_values_dict):
        pk_mm_lin = self._get_Pk_mm_lin()

        # now wish to interpolate sigma_R to these Rs
        zinterp, rinterp, sigma_r_interp = self.provider.get_sigma_R()
        sigma_rs = RectBivariateSpline(zinterp, rinterp, sigma_r_interp)

        output_pk_hm_mm = np.empty([len(self.z), len(self.k)])

        for iz, z_eval in enumerate(self.z):
            hmod = halo.model(
                z_eval,
                self.provider.get_param("omegam"),
                name=self.hmf_name,
                Dv=self.hmf_Dv,
            )

            lagrangian_radii = hmod.Lagrangian_radius(self.Ms)
            virial_radii = hmod.virial_radius(self.Ms)

            concentrations = 7.85 * (self.Ms / 2e12) ** -0.081 * (1.0 + z_eval) ** -0.71
            uk = self._win_NFW(self.k, virial_radii, concentrations)
            matter_profile = halo.profile.Fourier(
                self.k,
                self.Ms,
                uk,
                amplitude=self.Ms,
                normalisation=hmod.rhom,
                mass_tracer=True,
            )

            pk_2h, pk_1h, pk_hm = hmod.power_spectrum(
                self.k,
                pk_mm_lin[iz],
                self.Ms,
                sigma_rs(z_eval, lagrangian_radii)[0],
                {"m": matter_profile},
                verbose=False,
            )

            output_pk_hm_mm[iz] = pk_hm["m-m"]

        state["Pk_mm_grid"] = output_pk_hm_mm
        # state['Pk_gm_grid'] = pk_hm['g-m']
        # state['Pk_gg_grid'] = pk_hm['g-g']

    def _win_NFW(
        self,
        k: np.ndarray,
        virial_radius: np.ndarray,
        concentration: np.ndarray,
    ) -> np.ndarray:
        from scipy.special import sici

        rs = virial_radius / concentration
        kv = np.outer(k, virial_radius)
        ks = np.outer(k, rs)
        sisv, cisv = sici(ks + kv)
        sis, cis = sici(ks)
        f1 = np.cos(ks) * (cisv - cis)
        f2 = np.sin(ks) * (sisv - sis)
        f3 = np.sin(kv) / (ks + kv)
        f4 = np.log(1.0 + concentration) - concentration / (1.0 + concentration)
        wk = (f1 + f2 - f3) / f4
        return wk
