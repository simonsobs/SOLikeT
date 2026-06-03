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
    soliket.CCL:
    soliket.halo_model.HaloModel_ccl:


Implementing your own halo model
--------------------------------

If you want to add your own halo model, you can do so by inheriting from the
``soliket.HaloModel`` theory class and implementing your own custom ``calculate()``
function (have a look at the simple ``HaloModel_ccl`` model for ideas).
"""

import numpy as np
from cobaya.log import LoggedError
from cobaya.theory import Provider, Theory


class HaloModel(Theory):
    """Abstract parent class for implementing Halo Models."""

    kmax: int | float
    z: float | list[float] | np.ndarray
    extra_args: dict | None

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
            self.k, self.z, pk_mm = self.provider.get_Pk_grid(
                var_pair=pair, nonlinear=False
            )
        return pk_mm

    def get_Pk_mm_grid(self) -> np.ndarray:
        return self.current_state["Pk_mm_grid"]

    def get_Pk_gg_grid(self) -> np.ndarray:
        return self.current_state["Pk_gg_grid"]

    def get_Pk_gm_grid(self) -> np.ndarray:
        return self.current_state["Pk_gm_grid"]


class HaloModel_ccl(HaloModel):
    """Simple halo model for the non-linear matter power spectrum, built on the CCL
    halo-model framework (:mod:`pyccl.halos`).

    A minimal demonstration of how to obtain a halo-model matter power spectrum using
    CCL. The mass function, halo bias, concentration and mass definition are exposed as
    configuration options so the basic ingredients of a halo model can be varied. It is
    built on the CCL cosmology supplied by the :class:`soliket.ccl.CCL` theory, which
    already carries the linear power spectrum computed by the Boltzmann code, so the
    result is consistent with the rest of the pipeline.
    """

    mass_function: str
    halo_bias: str
    concentration: str
    mass_def: str
    Mmin: float
    Mmax: float
    nM: int
    sigma_kmax: float
    zmax: float

    def initialize(self):
        super().initialize()
        import pyccl as ccl

        # These ingredients depend only on configuration, not on cosmology, so build
        # them once here rather than on every calculate() call.
        try:
            self._mass_function = ccl.halos.MassFunc.from_name(self.mass_function)(
                mass_def=self.mass_def
            )
            self._halo_bias = ccl.halos.HaloBias.from_name(self.halo_bias)(
                mass_def=self.mass_def
            )
            concentration = ccl.halos.Concentration.from_name(self.concentration)(
                mass_def=self.mass_def
            )
        except (KeyError, ValueError) as e:
            raise LoggedError(
                self.log,
                f"Could not build the CCL halo model with mass_function="
                f"'{self.mass_function}', halo_bias='{self.halo_bias}', "
                f"concentration='{self.concentration}', mass_def='{self.mass_def}'. "
                f"Note that some concentration models are only defined for specific "
                f"mass definitions. Original error: {e}",
            ) from e
        self._profile = ccl.halos.HaloProfileNFW(
            mass_def=self.mass_def, concentration=concentration
        )
        self._hmc = ccl.halos.HMCalculator(
            mass_function=self._mass_function,
            halo_bias=self._halo_bias,
            mass_def=self.mass_def,
            log10M_min=np.log10(self.Mmin),
            log10M_max=np.log10(self.Mmax),
            nM=self.nM,
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
        # The halo model is only meaningful where haloes have collapsed; CCL's mass
        # function fails at very high z, so cap the sampling.
        self.z = self.z[self.z <= self.zmax]
        # CCL computes sigma(M) by integrating the linear power, so the spectrum must
        # extend well beyond the output kmax to converge for low-mass haloes.
        pk_kmax = max(self.kmax, self.sigma_kmax)
        return {
            "CCL": {"kmax": pk_kmax, "z": self.z, "nonlinear": False},
            "Pk_grid": {
                "vars_pairs": self._var_pairs,
                "nonlinear": False,
                "z": self.z,
                "k_max": pk_kmax,
            },
        }

    def calculate(self, state: dict, want_derived: bool = True, **params_values_dict):
        ccl = self.provider.get_CCL()["ccl"]
        cosmo = self.provider.get_CCL()["cosmo"]

        for pair in self._var_pairs:
            self.k, self.z, _ = self.provider.get_Pk_grid(var_pair=pair, nonlinear=False)

        output_pk_hm_mm = np.empty([len(self.z), len(self.k)])
        for iz, z_eval in enumerate(self.z):
            output_pk_hm_mm[iz] = ccl.halos.halomod_power_spectrum(
                cosmo, self._hmc, self.k, 1.0 / (1.0 + z_eval), self._profile
            )

        state["Pk_mm_grid"] = output_pk_hm_mm
