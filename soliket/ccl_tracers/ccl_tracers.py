"""
:Synopsis: Likelihood for cross-correlation of CMB lensing with Large Scale Structure
data. Makes use of the cobaya CCL module for handling tracers and Limber integration.

:Authors: Pablo Lemos, Ian Harrison.
"""

from typing import ClassVar

import numpy as np

try:
    from numpy import trapezoid
except ImportError:
    from numpy import trapz as trapezoid
from cobaya.log import LoggedError
from cobaya.theory import Provider

from soliket.ccl import CCL
from soliket.gaussian import GaussianLikelihood


class CCLTracersLikelihood(GaussianLikelihood):
    r"""
    Generic likelihood for CCL tracer objects.
    """

    datapath: str
    use_spectra: str | list[tuple[str, str]]
    ncovsims: int | None
    provider: Provider

    def initialize(self):
        super().initialize()

    def get_requirements(self) -> dict:
        return {"CCL": {"kmax": 10, "nonlinear": True}, "zstar": None}

    def _get_CCL_results(self) -> tuple[CCL, dict]:
        cosmo_dict = self.provider.get_CCL()
        return cosmo_dict["ccl"], cosmo_dict["cosmo"]

    def _get_nz(
        self, z: np.ndarray, tracer, tracer_name: str, **params_values
    ) -> np.ndarray:
        if self.z_nuisance_mode == "deltaz":
            bias = params_values[f"{tracer_name}_deltaz"]
            return tracer.get_dndz(z - bias)
        raise ValueError(f"Unknown z_nuisance_mode {self.z_nuisance_mode!r}")

    def _get_ia_bias(
        self,
        z_tracer: np.ndarray,
        nz_tracer: np.ndarray,
        tracer_name: str,
        params_values: dict,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if self.ia_mode is None:
            return None
        elif self.ia_mode == "nla":
            A_IA = params_values["A_IA"]
            eta_IA = params_values["eta_IA"]
            # n(z)-weighted mean redshift of this tracer, used as the NLA pivot.
            # CCL's WeakLensingTracer (use_A_ia=True, the default) supplies the
            # C1·rho_crit·Omega_m/D(z) normalization (Joachimi 2011 Eq. 6), so we
            # pass only the dimensionless ((1+z)/(1+z0))^eta evolution shape.
            z0_IA = trapezoid(z_tracer * nz_tracer, x=z_tracer) / trapezoid(
                nz_tracer, x=z_tracer
            )
            return (z_tracer, A_IA * ((1 + z_tracer) / (1 + z0_IA)) ** eta_IA)
        elif self.ia_mode == "nla-perbin":
            A_IA = params_values[f"{tracer_name}_A_IA"]
            return (z_tracer, A_IA * np.ones_like(z_tracer))
        elif self.ia_mode == "nla-noevo":
            A_IA = params_values["A_IA"]
            return (z_tracer, A_IA * np.ones_like(z_tracer))
        return None


class CCLTracersCrossLikelihood(CCLTracersLikelihood):
    r"""
    Generic likelihood for cross-correlations of CCL tracer objects.
    """

    datapath: str
    use_spectra: str | list[tuple[str, str]]
    ncovsims: int | None
    provider: Provider

    # Physical quantities :meth:`_get_tracer` knows how to build. The base
    # ``_check_tracers`` restricts the SACC data to ``_allowable_tracers``;
    # ``_check_buildable_tracers`` then guarantees ``_allowable_tracers`` itself
    # stays within this set, so every quantity reaching ``_get_tracer`` is buildable.
    _BUILDABLE_QUANTITIES: ClassVar[list[str]] = [
        "cmb_convergence",
        "galaxy_density",
        "galaxy_shear",
    ]

    def initialize(self):
        super().initialize()
        self._check_is_cross()
        self._check_buildable_tracers()

    def _check_buildable_tracers(self):
        """Reject (at init) any allowed quantity ``_get_tracer`` cannot build.

        Catches the developer-side mismatch where a subclass lists a quantity in
        ``_allowable_tracers`` that ``_get_tracer`` has no branch for -- regardless
        of whether the current SACC data happens to exercise it.
        """
        unbuildable = set(self._allowable_tracers or ()) - set(self._BUILDABLE_QUANTITIES)
        if unbuildable:
            raise LoggedError(
                self.log,
                f"{self.__class__.__name__} allows tracer quantities "
                f"{sorted(unbuildable)} that it cannot build; _get_tracer supports "
                f"only {list(self._BUILDABLE_QUANTITIES)}.",
            )

    def _check_is_cross(self):
        for tracer_comb in self.sacc_data.get_tracer_combinations():
            if (
                self.sacc_data.tracers[tracer_comb[0]].quantity
                == self.sacc_data.tracers[tracer_comb[1]].quantity
            ):
                raise LoggedError(
                    self.log,
                    "You have tried to use {} to calculate an \
                                   autocorrelation, but it is a cross-correlation \
                                   likelihood. Please check your tracer selection in the \
                                   ini file.".format(self.__class__.__name__),
                )

    def _get_tracer(self, ccl: CCL, cosmo: dict, tracer_name: str, params_values: dict):
        """Build the CCL tracer for a SACC tracer, by its physical quantity."""
        quantity = self.sacc_data.tracers[tracer_name].quantity
        if quantity == "cmb_convergence":
            return ccl.CMBLensingTracer(cosmo, z_source=self.provider.get_param("zstar"))

        z = self.sacc_data.tracers[tracer_name].z
        nz = self.sacc_data.tracers[tracer_name].nz
        if quantity == "galaxy_density":
            return ccl.NumberCountsTracer(
                cosmo,
                has_rsd=False,
                dndz=(z, nz),
                bias=(z, params_values["b1"] * np.ones(len(z))),
                mag_bias=(z, params_values["s1"] * np.ones(len(z))),
            )
        if quantity == "galaxy_shear":
            ia_z = self._get_ia_bias(z, nz, tracer_name, params_values)
            tracer = ccl.WeakLensingTracer(cosmo, dndz=(z, nz), ia_bias=ia_z)
            if getattr(self, "z_nuisance_mode", None) is not None:
                nz = self._get_nz(z, tracer, tracer_name, **params_values)
                tracer = ccl.WeakLensingTracer(cosmo, dndz=(z, nz), ia_bias=ia_z)
            return tracer
        raise ValueError(
            f"Tracer {tracer_name!r} has unsupported quantity {quantity!r}; "
            f"{self.__class__.__name__} can build tracers only for "
            f"{list(self._BUILDABLE_QUANTITIES)}."
        )

    def _get_unbinned_theory(self, **params_values) -> list[np.ndarray]:
        """Unbinned Limber spectra per tracer combination (binning done by base).

        Shared by all CCL cross-correlation likelihoods: build both tracers from
        their physical quantity, take the Limber ``angular_cl`` on each tracer
        pair's bandpower-window support, and apply shear multiplicative bias.
        """
        ccl, cosmo = self._get_CCL_results()
        cl_unbinned_list: list[np.ndarray] = []

        for tracer_comb in self.sacc_data.get_tracer_combinations():
            tracer1 = self._get_tracer(ccl, cosmo, tracer_comb[0], params_values)
            tracer2 = self._get_tracer(ccl, cosmo, tracer_comb[1], params_values)
            ells_theory, _ = self.get_binning(tracer_comb)

            cl_unbinned = ccl.cells.angular_cl(cosmo, tracer1, tracer2, ells_theory)

            shear_name = next(
                (
                    name
                    for name in tracer_comb
                    if self.sacc_data.tracers[name].quantity == "galaxy_shear"
                ),
                None,
            )
            if shear_name is not None and (
                getattr(self, "m_nuisance_mode", None) is not None
            ):
                # note shear x shear (both tracers) is not handled here
                cl_unbinned = (1 + params_values[f"{shear_name}_m"]) * cl_unbinned

            cl_unbinned_list.append(cl_unbinned)
        return cl_unbinned_list


class CCLTracersAutoLikelihood(CCLTracersLikelihood):
    r"""
    Generic likelihood for auto-correlations of CCL tracer objects.
    """

    datapath: str
    use_spectra: str | list[tuple[str, str]]
    ncovsims: int | None
    provider: Provider

    def initialize(self):
        super().initialize()
        self._check_is_auto()

    def _check_is_auto(self):
        for tracer_comb in self.sacc_data.get_tracer_combinations():
            if not (
                self.sacc_data.tracers[tracer_comb[0]].quantity
                == self.sacc_data.tracers[tracer_comb[1]].quantity
            ):
                raise LoggedError(
                    self.log,
                    "You have tried to use {} to calculate a \
                                   cross-correlation, but it is an auto-correlation \
                                   likelihood. Please check your tracer selection in the \
                                   ini file.".format(self.__class__.__name__),
                )


class GalaxyKappaLikelihood(CCLTracersCrossLikelihood):
    r"""
    Likelihood for cross-correlations of galaxy and CMB lensing data.
    """

    name: str = "GalaxyKappa"
    _allowable_tracers: ClassVar[list[str]] = ["cmb_convergence", "galaxy_density"]
    params: dict

    # Theory comes from the shared CCLTracersCrossLikelihood._get_unbinned_theory.


class ShearKappaLikelihood(CCLTracersCrossLikelihood):
    r"""
    Likelihood for cross-correlations of galaxy weak lensing shear and CMB lensing data.
    """

    name: str = "ShearKappa"
    _allowable_tracers: ClassVar[list[str]] = ["cmb_convergence", "galaxy_shear"]

    z_nuisance_mode: str | bool | None
    m_nuisance_mode: str | bool | None
    ia_mode: str | None
    params: dict

    # Theory comes from the shared CCLTracersCrossLikelihood._get_unbinned_theory,
    # which builds shear tracers (with IA / redshift nuisance) and applies the
    # multiplicative-bias nuisance.
