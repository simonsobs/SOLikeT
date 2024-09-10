"""
:Synopsis: Likelihood for cross-correlation of CMB lensing with Large Scale Structure
data. Makes use of the cobaya CCL module for handling tracers and Limber integration.

:Authors: Pablo Lemos, Ian Harrison.
"""

from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union
import numpy as np


try:
    from numpy import trapezoid
except ImportError:
    from numpy import trapz as trapezoid
import sacc
from cobaya.log import LoggedError
from cobaya.theory import Provider

from soliket.ccl import CCL
from soliket.gaussian import GaussianData, GaussianLikelihood


class CrossCorrelationLikelihood(GaussianLikelihood):
    r"""
    Generic likelihood for cross-correlations of CCL tracer objects.
    """

    datapath: str
    use_spectra: Union[str, List[Tuple[str, str]]]
    ncovsims: Optional[int]
    provider: Provider

    def initialize(self):

        self._get_sacc_data()
        self._check_tracers()

    def get_requirements(self) -> Dict[str, dict]:
        return {"CCL": {"kmax": 10, "nonlinear": True}, "zstar": None}

    def _get_CCL_results(self) -> Tuple[CCL, dict]:
        cosmo_dict = self.provider.get_CCL()
        return cosmo_dict["ccl"], cosmo_dict["cosmo"]

    def _check_tracers(self) -> None:

        # check correct tracers
        for tracer_comb in self.sacc_data.get_tracer_combinations():

            if (self.sacc_data.tracers[tracer_comb[0]].quantity ==
                    self.sacc_data.tracers[tracer_comb[1]].quantity):
                raise LoggedError(self.log,
                                  f'You have tried to use {self.__class__.__name__} \
                                    to calculate an autocorrelation, but it is a \
                                    cross-correlation likelihood. Please check your \
                                    tracer selection in the ini file.')

            for tracer in tracer_comb:
                if self.sacc_data.tracers[tracer].quantity not in self._allowable_tracers:
                    raise LoggedError(
                        self.log,
                        f'You have tried to use a \
                        {self.sacc_data.tracers[tracer].quantity} tracer in \
                        {self.__class__.__name__}, which only allows \
                        {self._allowable_tracers}. Please check your \
                        tracer selection in the ini file.'
                        )

    def _get_nz(
        self,
        z: np.ndarray,
        tracer: Any,
        tracer_name: str,
        **params_values: dict
    ) -> np.ndarray:
        if self.z_nuisance_mode == 'deltaz':
            bias = params_values[f'{tracer_name}_deltaz']
            nz_biased = tracer.get_dndz(z - bias)

        # nz_biased /= np.trapezoid(nz_biased, z)

        return nz_biased

    def _get_sacc_data(self, **params_values: dict) -> None:
        self.sacc_data = sacc.Sacc.load_fits(self.datapath)

        if self.use_spectra == 'all':
            pass
        else:
            for tracer_comb in self.sacc_data.get_tracer_combinations():
                if tracer_comb not in self.use_spectra:
                    self.sacc_data.remove_selection(tracers=tracer_comb)

        self.x = self._construct_ell_bins()
        self.y = self.sacc_data.mean
        self.cov = self.sacc_data.covariance.covmat

        self.data = GaussianData(self.name, self.x, self.y, self.cov, self.ncovsims)

    def _construct_ell_bins(self) -> np.ndarray:
        ell_eff = []

        for tracer_comb in self.sacc_data.get_tracer_combinations():
            ind = self.sacc_data.indices(tracers=tracer_comb)
            ell = np.array(self.sacc_data._get_tags_by_index(["ell"], ind)[0])
            ell_eff.append(ell)

        return np.concatenate(ell_eff)

    def _get_data(
        self, **params_values: dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        data_auto = np.loadtxt(self.auto_file)
        data_cross = np.loadtxt(self.cross_file)

        # Get data
        self.ell_auto = data_auto[0]
        cl_auto = data_auto[1]
        cl_auto_err = data_auto[2]

        self.ell_cross = data_cross[0]
        cl_cross = data_cross[1]
        cl_cross_err = data_cross[2]

        x = np.concatenate([self.ell_auto, self.ell_cross])
        y = np.concatenate([cl_auto, cl_cross])
        dy = np.concatenate([cl_auto_err, cl_cross_err])

        return x, y, dy

    def get_binning(self, tracer_comb: tuple) -> Tuple[np.ndarray, np.ndarray]:
        bpw_idx = self.sacc_data.indices(tracers=tracer_comb)
        bpw = self.sacc_data.get_bandpower_windows(bpw_idx)
        ells_theory = bpw.values
        ells_theory = np.asarray(ells_theory, dtype=int)
        w_bins = bpw.weight.T

        return ells_theory, w_bins

    def logp(self, **params_values: dict) -> float:
        theory = self._get_theory(**params_values)
        return self.data.loglike(theory)


class GalaxyKappaLikelihood(CrossCorrelationLikelihood):
    r"""
    Likelihood for cross-correlations of galaxy and CMB lensing data.
    """
    _allowable_tracers: ClassVar[List[str]] = ['cmb_convergence', 'galaxy_density']
    params: dict

    def _get_theory(self, **params_values: dict) -> np.ndarray:
        ccl, cosmo = self._get_CCL_results()

        tracer_comb = self.sacc_data.get_tracer_combinations()

        for tracer in np.unique(tracer_comb):
            if self.sacc_data.tracers[tracer].quantity == "cmb_convergence":
                cmbk_tracer = tracer
            elif self.sacc_data.tracers[tracer].quantity == "galaxy_density":
                gal_tracer = tracer

        z_gal_tracer = self.sacc_data.tracers[gal_tracer].z
        nz_gal_tracer = self.sacc_data.tracers[gal_tracer].nz

        # this should use the bias theory!
        tracer_g = ccl.NumberCountsTracer(
            cosmo,
            has_rsd=False,
            dndz=(z_gal_tracer, nz_gal_tracer),
            bias=(z_gal_tracer, params_values["b1"] * np.ones(len(z_gal_tracer))),
            mag_bias=(z_gal_tracer, params_values["s1"] * np.ones(len(z_gal_tracer))),
        )
        tracer_k = ccl.CMBLensingTracer(cosmo, z_source=self.provider.get_param('zstar'))

        ells_theory_gk, w_bins_gk = self.get_binning((gal_tracer, cmbk_tracer))

        cl_gk_unbinned = ccl.cells.angular_cl(cosmo, tracer_k, tracer_g, ells_theory_gk)

        cl_gk_binned = np.dot(w_bins_gk, cl_gk_unbinned)

        return cl_gk_binned


class ShearKappaLikelihood(CrossCorrelationLikelihood):
    r"""
    Likelihood for cross-correlations of galaxy weak lensing shear and CMB lensing data.
    """
    _allowable_tracers: ClassVar[List[str]] = ["cmb_convergence", "galaxy_shear"]

    z_nuisance_mode: Optional[Union[str, bool]]
    m_nuisance_mode: Optional[Union[str, bool]]
    ia_mode: Optional[str]
    params: dict

    def _get_theory(self, **params_values: dict) -> np.ndarray:
        ccl, cosmo = self._get_CCL_results()
        cl_binned_list: List[np.ndarray] = []

        for tracer_comb in self.sacc_data.get_tracer_combinations():
            tracer1 = self._get_tracer(ccl, cosmo, tracer_comb[0], params_values)
            tracer2 = self._get_tracer(ccl, cosmo, tracer_comb[1], params_values)

            bpw_idx = self.sacc_data.indices(tracers=tracer_comb)
            bpw = self.sacc_data.get_bandpower_windows(bpw_idx)
            ells_theory = np.asarray(bpw.values, dtype=int)
            w_bins = bpw.weight.T

            cl_unbinned = ccl.cells.angular_cl(cosmo, tracer1, tracer2, ells_theory)
            if self.m_nuisance_mode is not None:
                sheartracer_name = (
                    tracer_comb[1]
                    if self.sacc_data.tracers[tracer_comb[1]].quantity == "galaxy_shear"
                    else tracer_comb[0]
                )
                m_bias = params_values[f'{sheartracer_name}_m']
                cl_unbinned = (1 + m_bias) * cl_unbinned

            cl_binned = np.dot(w_bins, cl_unbinned)
            cl_binned_list.append(cl_binned)

        cl_binned_total = np.concatenate(cl_binned_list)
        return cl_binned_total

    def _get_tracer(
        self, ccl: CCL, cosmo: dict, tracer_name: str, params_values: dict
    ):
        tracer_data = self.sacc_data.tracers[tracer_name]
        if tracer_data.quantity == "cmb_convergence":
            return ccl.CMBLensingTracer(cosmo, z_source=self.provider.get_param('zstar'))
        elif tracer_data.quantity == "galaxy_shear":
            z_tracer = self.sacc_data.tracers[tracer_name].z
            nz_tracer = self.sacc_data.tracers[tracer_name].nz
            ia_z = self._get_ia_bias(z_tracer, nz_tracer, tracer_name, params_values)

            tracer = ccl.WeakLensingTracer(
                cosmo, dndz=(z_tracer, nz_tracer), ia_bias=ia_z
            )
            if self.z_nuisance_mode is not None:
                nz_tracer = self._get_nz(z_tracer, tracer, tracer_name, **params_values)
                tracer = ccl.WeakLensingTracer(
                    cosmo, dndz=(z_tracer, nz_tracer), ia_bias=ia_z
                )
            return tracer
        return None


    def _get_ia_bias(
        self,
        z_tracer: np.ndarray,
        nz_tracer: np.ndarray,
        tracer_name: str,
        params_values: dict
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.ia_mode is None:
            return None
        elif self.ia_mode == 'nla':
            A_IA = params_values['A_IA']
            eta_IA = params_values['eta_IA']
            z0_IA = trapezoid(z_tracer * nz_tracer)
            return (z_tracer, A_IA * ((1 + z_tracer) / (1 + z0_IA)) ** eta_IA)
        elif self.ia_mode == 'nla-perbin':
            A_IA = params_values[f'{tracer_name}_A_IA']
            return (z_tracer, A_IA * np.ones_like(z_tracer))
        elif self.ia_mode == 'nla-noevo':
            A_IA = params_values['A_IA']
            return (z_tracer, A_IA * np.ones_like(z_tracer))
        return None
