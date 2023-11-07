"""
:Synopsis: Likelihood for cross-correlation of CMB lensing with Large Scale Structure 
data. Makes use of the cobaya CCL module for handling tracers and Limber integration.

:Authors: Pablo Lemos, Ian Harrison.
"""

import numpy as np
from ..gaussian import GaussianData, GaussianLikelihood
import pyccl as ccl
from cobaya.log import LoggedError
import pyccl.nl_pt as pt

import sacc


class CrossCorrelationLikelihood(GaussianLikelihood):
    r"""
    Generic likelihood for cross-correlations of CCL tracer objects.
    """
    def initialize(self):

        self._get_sacc_data()
        self._check_tracers()

    def get_requirements(self):
        return {"CCL": {"kmax": 10, "nonlinear": True}}

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

        self.twopoints = self.sacc_data.get_tracer_combinations()

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

    def initialize(self):

        self.PT_bias = self.bz_model in ['LagrangianPT', 'EulerianPT', 'BACCO', 'anzu']

        self._get_sacc_data()
        self._check_tracers()
        self._initialize_pt()

    def _get_sacc_data(self, **params_values):

        self.sacc_data = sacc.Sacc.load_fits(self.datapath)

        if self.use_spectra == 'all':
            pass
        else:
            for tracer_comb in self.sacc_data.get_tracer_combinations():
                if tracer_comb not in self.use_spectra:
                    self.sacc_data.remove_selection(tracers=tracer_comb)

        self.twopoints = self.sacc_data.get_tracer_combinations()
        self.bin_properties = {}
        for b in self.bins:
            if b not in self.sacc_data.tracers:
                raise LoggedError(self.log, "Unknown tracer %s" % b)
            t = self.sacc_data.tracers[b]
            if t.quantity != 'cmb_convergence':
                zmean = np.average(t.z, weights=t.nz)
                self.bin_properties[b] = {'z_fid': t.z,
                                          'nz_fid': t.nz,
                                          'zmean_fid': zmean}

        self.x = self._construct_ell_bins()
        self.y = self.sacc_data.mean
        self.cov = self.sacc_data.covariance.covmat

        self.data = GaussianData(self.name, self.x, self.y, self.cov, self.ncovsims)

    def _initialize_pt(self):

        if self.bz_model == 'LagrangianPT':
            self.ptc = pt.LagrangianPTCalculator(log10k_min=-4, log10k_max=2, nk_per_decade=20)
        elif self.bz_model == 'EulerianPT':
            self.ptc = pt.EulerianPTCalculator(with_NC=True, with_IA=True, log10k_min=-4,
                                               log10k_max=2, nk_per_decade=20)

    def _get_nz(self, tr_name, **pars):
        z = self.bin_properties[tr_name]['z_fid']
        nz = self.bin_properties[tr_name]['nz_fid']
        if self.nz_model == 'NzShift':
            z = z + pars[self.input_params_prefix + '_' + tr_name + '_dz']
            msk = z >= 0
            z = z[msk]
            nz = nz[msk]
        elif self.nz_model != 'NzNone':
            raise LoggedError(self.log, "Unknown Nz model %s" % self.nz_model)
        return (z, nz)

    def _get_bz(self, tr_name, **pars):
        """ Get linear galaxy bias. Unless we're using a linear bias,
        this should be just 1."""
        z = self.bin_properties[tr_name]['z_fid']
        zmean = self.bin_properties[tr_name]['zmean_fid']
        bz = np.ones_like(z)

        if self.bz_model == 'linear':
            b1 = pars[self.input_params_prefix + '_' + tr_name + '_b1']
            b1p = pars[self.input_params_prefix + '_' + tr_name + '_b1p']
            bz = b1 + b1p * (z - zmean)
        return (z, bz)

    def _get_tracers(self, cosmo, **params_values):

        trs = {}

        for tr_name in np.unique(self.twopoints):
            q = self.sacc_data[tr_name].quantity
            trs[tr_name] = {}
            if q == 'galaxy_density':
                nz = self._get_nz(tr_name, **params_values)
                bz = self._get_bz(tr_name, **params_values)
                t = ccl.NumberCountsTracer(cosmo, dndz=nz, bias=bz, has_rsd=False)
                if self.PT_bias:
                    z = self.bin_properties[tr_name]['z_fid']
                    zmean = self.bin_properties[tr_name]['zmean_fid']

                    pref = self.input_params_prefix + '_' + tr_name

                    b1 = params_values[pref + '_b1']
                    b1p = params_values.get(pref + '_b1p', None)
                    if b1p is not None and b1p != 0.:
                        b1z = b1 + b1p * (z - zmean)
                        b1 = (z, b1z)
                    b2 = params_values[pref + '_b2']
                    bs = params_values.get(pref + '_bs', None)
                    bk2 = params_values.get(pref + '_bk2', None)
                    b3nl = params_values.get(pref + '_b3nl', None)

                    ptt = pt.PTNumberCountsTracer(b1=b1, b2=b2, bs=bs, bk2=bk2, b3nl=b3nl)
            elif q == 'cmb_convergence':
                t = ccl.CMBLensingTracer(cosmo, z_source=self.z_cmb)
                if self.PT_bias:
                    ptt = pt.PTMatterTracer()
            trs[tr_name]['ccl_tracer'] = t
            if self.PT_bias:
                trs[tr_name]['PT_tracer'] = ptt

        return trs

    def _get_theory(self, **params_values):

        cosmo = self.provider.get_CCL()["cosmo"]
        self.ptc.update_ingredients(cosmo)
        tracers = self._get_tracers(cosmo, **params_values)

        cls = []

        for tr_pair in self.twopoints:
            tr_x, tr_y = tr_pair

            ells_theory, w_bins = self.get_binning((tr_x, tr_y))

            if self.PT_bias:
                pk_xy = self.ptc.get_biased_pk2d(tracers[tr_x]['PT_tracer'], tracer2=tracers[tr_y]['PT_tracer'])
                cl_unbinned = ccl.cells.angular_cl(cosmo, tracers[tr_x]['CCL_tracer'], tracers[tr_y]['CCL_tracer'],
                                                      ells_theory, p_of_k_a=pk_xy)
            else:
                cl_unbinned = ccl.cells.angular_cl(cosmo, tracers[tr_x]['CCL_tracer'], tracers[tr_y]['CCL_tracer'],
                                                   ells_theory)

            cl_binned = np.dot(w_bins, cl_unbinned)

            cls.append(cl_binned)

        return cls


class ShearKappaLikelihood(CrossCorrelationLikelihood):
    r"""
    Likelihood for cross-correlations of galaxy weak lensing shear and CMB lensing data.
    """
    _allowable_tracers = ["cmb_convergence", "galaxy_shear"]

    def _get_theory(self, **params_values):

        cosmo = self.provider.get_CCL()["cosmo"]

        cl_binned_list = []

        for tracer_comb in self.sacc_data.get_tracer_combinations():

            if self.sacc_data.tracers[tracer_comb[0]].quantity == "cmb_convergence":
                tracer1 = ccl.CMBLensingTracer(cosmo, z_source=1060)

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

                    ia_z = (z_tracer1, A_IA * ((1 + z_tracer1) / (1 + z0_IA))**eta_IA)
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
                tracer2 = ccl.CMBLensingTracer(cosmo, z_source=1060)

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

                    ia_z = (z_tracer2, A_IA * ((1 + z_tracer2) / (1 + z0_IA))**eta_IA)
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
