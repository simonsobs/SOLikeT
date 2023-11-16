"""
:Synopsis: Likelihood for cross-correlation of CMB lensing with Large Scale Structure
data. Makes use of the cobaya CCL module for handling tracers and Limber integration.

:Authors: Pablo Lemos, Ian Harrison.
"""

import numpy as np
from ..gaussian import GaussianData, GaussianLikelihood
from cobaya.log import LoggedError

import sacc


class CrossCorrelationLikelihood(GaussianLikelihood):
    r"""
    Generic likelihood for cross-correlations of CCL tracer objects.
    """

    def initialize(self):

        self._get_sacc_data()
        self._check_tracers()

    def get_requirements(self):
        return {"CCL": {"kmax": 10, "nonlinear": True}, "zstar": None}

    def _get_CCL_results(self):
        cosmo_dict = self.provider.get_CCL()
        return cosmo_dict["ccl"], cosmo_dict["cosmo"]

    def _check_tracers(self):

        # check correct tracers
        for tracer_comb in self.sacc_data.get_tracer_combinations():

            if (self.sacc_data.tracers[tracer_comb[0]].quantity ==
                    self.sacc_data.tracers[tracer_comb[1]].quantity):
                raise LoggedError(self.log,
                                  'You have tried to use {0} to calculate an \
                                   autocorrelation, but it is a cross-correlation \
                                   likelihood. Please check your tracer selection in the \
                                   ini file.'.format(self.__class__.__name__))

            for tracer in tracer_comb:
                if self.sacc_data.tracers[tracer].quantity not in self._allowable_tracers:
                    raise LoggedError(self.log,
                                      'You have tried to use a {0} tracer in \
                                       {1}, which only allows {2}. Please check your \
                                       tracer selection in the ini file.\
                                       '.format(self.sacc_data.tracers[tracer].quantity,
                                                self.__class__.__name__,
                                                self._allowable_tracers))

    def _get_nz(self, z, tracer, tracer_name, **params_values):

        if self.z_nuisance_mode == 'deltaz':
            bias = params_values['{}_deltaz'.format(tracer_name)]
            nz_biased = tracer.get_dndz(z - bias)

        # nz_biased /= np.trapz(nz_biased, z)

        return nz_biased

    def _get_sacc_data(self, **params_values):

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

    def _construct_ell_bins(self):

        ell_eff = []

        for tracer_comb in self.sacc_data.get_tracer_combinations():
            ind = self.sacc_data.indices(tracers=tracer_comb)
            ell = np.array(self.sacc_data._get_tags_by_index(["ell"], ind)[0])
            ell_eff.append(ell)

        return np.concatenate(ell_eff)

    def _get_data(self, **params_values):
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

    def get_binning(self, tracer_comb):

        bpw_idx = self.sacc_data.indices(tracers=tracer_comb)
        bpw = self.sacc_data.get_bandpower_windows(bpw_idx)
        ells_theory = bpw.values
        ells_theory = np.asarray(ells_theory, dtype=int)
        w_bins = bpw.weight.T

        return ells_theory, w_bins

    def logp(self, **params_values):
        theory = self._get_theory(**params_values)
        return self.data.loglike(theory)


class GalaxyKappaLikelihood(CrossCorrelationLikelihood):
    r"""
    Likelihood for cross-correlations of galaxy and CMB lensing data.
    """
    _allowable_tracers = ['cmb_convergence', 'galaxy_density']

    def _get_theory(self, **params_values):

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
        tracer_g = ccl.NumberCountsTracer(cosmo,
                                          has_rsd=False,
                                          dndz=(z_gal_tracer, nz_gal_tracer),
                                          bias=(z_gal_tracer,
                                                params_values["b1"] *
                                                np.ones(len(z_gal_tracer))),
                                          mag_bias=(z_gal_tracer,
                                                    params_values["s1"] *
                                                    np.ones(len(z_gal_tracer)))
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
    _allowable_tracers = ["cmb_convergence", "galaxy_shear"]

    def _get_theory(self, **params_values):

        ccl, cosmo = self._get_CCL_results()

        cl_binned_list = []

        for tracer_comb in self.sacc_data.get_tracer_combinations():

            if self.sacc_data.tracers[tracer_comb[0]].quantity == "cmb_convergence":
                tracer1 = ccl.CMBLensingTracer(cosmo,
                                               z_source=self.provider.get_param('zstar'))

            elif self.sacc_data.tracers[tracer_comb[0]].quantity == "galaxy_shear":

                sheartracer_name = tracer_comb[0]

                z_tracer1 = self.sacc_data.tracers[tracer_comb[0]].z
                nz_tracer1 = self.sacc_data.tracers[tracer_comb[0]].nz

                if self.ia_mode is None:
                    ia_z = None
                elif self.ia_mode == 'nla':
                    A_IA = params_values['A_IA']
                    eta_IA = params_values['eta_IA']
                    z0_IA = np.trapz(z_tracer1 * nz_tracer1)

                    ia_z = (z_tracer1, A_IA * ((1 + z_tracer1) / (1 + z0_IA)) ** eta_IA)
                elif self.ia_mode == 'nla-perbin':
                    A_IA = params_values['{}_A_IA'.format(sheartracer_name)]
                    ia_z = (z_tracer1, A_IA * np.ones_like(z_tracer1))
                elif self.ia_mode == 'nla-noevo':
                    A_IA = params_values['A_IA']
                    ia_z = (z_tracer1, A_IA * np.ones_like(z_tracer1))

                tracer1 = ccl.WeakLensingTracer(cosmo,
                                                dndz=(z_tracer1, nz_tracer1),
                                                ia_bias=ia_z)

                if self.z_nuisance_mode is not None:
                    nz_tracer1 = self._get_nz(z_tracer1,
                                              tracer1,
                                              tracer_comb[0],
                                              **params_values)

                    tracer1 = ccl.WeakLensingTracer(cosmo,
                                                    dndz=(z_tracer1, nz_tracer1),
                                                    ia_bias=ia_z)

            if self.sacc_data.tracers[tracer_comb[1]].quantity == "cmb_convergence":
                tracer2 = ccl.CMBLensingTracer(cosmo,
                                               z_source=self.provider.get_param('zstar'))

            elif self.sacc_data.tracers[tracer_comb[1]].quantity == "galaxy_shear":

                sheartracer_name = tracer_comb[1]

                z_tracer2 = self.sacc_data.tracers[tracer_comb[1]].z
                nz_tracer2 = self.sacc_data.tracers[tracer_comb[1]].nz

                if self.ia_mode is None:
                    ia_z = None
                elif self.ia_mode == 'nla':
                    A_IA = params_values['A_IA']
                    eta_IA = params_values['eta_IA']
                    z0_IA = np.trapz(z_tracer2 * nz_tracer2)

                    ia_z = (z_tracer2, A_IA * ((1 + z_tracer2) / (1 + z0_IA)) ** eta_IA)
                elif self.ia_mode == 'nla-perbin':
                    A_IA = params_values['{}_A_IA'.format(sheartracer_name)]
                    ia_z = (z_tracer2, A_IA * np.ones_like(z_tracer2))
                elif self.ia_mode == 'nla-noevo':
                    A_IA = params_values['A_IA']
                    ia_z = (z_tracer2, A_IA * np.ones_like(z_tracer2))

                tracer2 = ccl.WeakLensingTracer(cosmo,
                                                dndz=(z_tracer2, nz_tracer2),
                                                ia_bias=ia_z)

                if self.z_nuisance_mode is not None:
                    nz_tracer2 = self._get_nz(z_tracer2,
                                              tracer2,
                                              tracer_comb[1],
                                              **params_values)

                    tracer2 = ccl.WeakLensingTracer(cosmo,
                                                    dndz=(z_tracer2, nz_tracer2),
                                                    ia_bias=ia_z)

            bpw_idx = self.sacc_data.indices(tracers=tracer_comb)
            bpw = self.sacc_data.get_bandpower_windows(bpw_idx)
            ells_theory = bpw.values
            ells_theory = np.asarray(ells_theory, dtype=int)
            w_bins = bpw.weight.T

            cl_unbinned = ccl.cells.angular_cl(cosmo, tracer1, tracer2, ells_theory)

            if self.m_nuisance_mode is not None:
                # note this allows wrong calculation, as we can do
                # shear x shear if the spectra are in the sacc
                # but then we would want (1 + m1) * (1 + m2)
                m_bias = params_values['{}_m'.format(sheartracer_name)]
                cl_unbinned = (1 + m_bias) * cl_unbinned

            cl_binned = np.dot(w_bins, cl_unbinned)

            cl_binned_list.append(cl_binned)

        cl_binned_total = np.concatenate(cl_binned_list)

        return cl_binned_total
