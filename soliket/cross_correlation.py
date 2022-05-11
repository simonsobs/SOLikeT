"""
Simple likelihood for CMB-galaxy cross-correlations using the cobaya
CCL module.

First version by Pablo Lemos
"""

import numpy as np
from .gaussian import GaussianData, GaussianLikelihood
import pyccl as ccl
from cobaya.log import LoggedError

import sacc


class CrossCorrelationLikelihood(GaussianLikelihood):
    def initialize(self):

        if self.datapath is None:
            self.dndz = np.loadtxt(self.dndz_file)

            x, y, dy = self._get_data()
            cov = np.diag(dy**2)
            self.data = GaussianData("CrossCorrelation", x, y, cov)
        else:
            self._get_sacc_data()

    def get_requirements(self):
        return {"CCL": {"kmax": 10, "nonlinear": True}}

    def _get_sacc_data(self, **params_values):

        self.sacc_data = sacc.Sacc.load_fits(self.datapath)

        if self.use_tracers == 'all':
            pass
        else:
            raise LoggedError('Tracer selection not implemented yet!')
            # self.sacc_data.keep_selection(tracers=self.use_tracers.split(','))

        self.x = self._construct_ell_bins()
        self.y = self.sacc_data.mean
        self.cov = self.sacc_data.covariance.covmat

        self.data = GaussianData(self.name, self.x, self.y, self.cov)

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

    def logp(self, **params_values):
        theory = self._get_theory(**params_values)
        return self.data.loglike(theory)


class GalaxyKappaLikelihood(CrossCorrelationLikelihood):
    def _get_theory(self, **params_values):
        cosmo = self.provider.get_CCL()["cosmo"]

        tracer_g = ccl.NumberCountsTracer(cosmo,
                                          has_rsd=False,
                                          dndz=self.dndz.T,
                                          bias=(self.dndz[:, 0],
                                                params_values["b1"] *
                                                    np.ones(len(self.dndz[:, 0]))),
                                          mag_bias=(self.dndz[:, 0],
                                                    params_values["s1"] *
                                                        np.ones(len(self.dndz[:, 0])))
                                          )
        tracer_k = ccl.CMBLensingTracer(cosmo, z_source=1060)

        cl_gg = ccl.cls.angular_cl(cosmo, tracer_g, tracer_g, self.ell_auto)  # + 1e-7
        cl_kg = ccl.cls.angular_cl(cosmo, tracer_k, tracer_g, self.ell_cross)

        return np.concatenate([cl_gg, cl_kg])


class ShearKappaLikelihood(CrossCorrelationLikelihood):

    def _get_theory(self, **params_values):

        cosmo = self.provider.get_CCL()["cosmo"]

        cl_binned_list = []

        for tracer_comb in self.sacc_data.get_tracer_combinations():

            if self.sacc_data.tracers[tracer_comb[0]].quantity == "cmb_convergence":
                tracer1 = ccl.CMBLensingTracer(cosmo, z_source=1060)
            elif self.sacc_data.tracers[tracer_comb[0]].quantity == "galaxy_shear":
                tracer1 = ccl.WeakLensingTracer(cosmo,
                                                dndz=(
                                                self.sacc_data.tracers[tracer_comb[0]].z,
                                                self.sacc_data.tracers[tracer_comb[0]].nz,
                                                ),
                                                ia_bias=None)

            if self.sacc_data.tracers[tracer_comb[1]].quantity == "cmb_convergence":
                tracer2 = ccl.CMBLensingTracer(cosmo, z_source=1060)
            elif self.sacc_data.tracers[tracer_comb[1]].quantity == "galaxy_shear":
                tracer2 = ccl.WeakLensingTracer(cosmo,
                                                dndz=(
                                                self.sacc_data.tracers[tracer_comb[1]].z,
                                                self.sacc_data.tracers[tracer_comb[1]].nz,
                                                ),
                                                ia_bias=None)

            bpw_idx = self.sacc_data.indices(tracers=tracer_comb)
            bpw = self.sacc_data.get_bandpower_windows(bpw_idx)
            ells_theory = bpw.values
            w_bins = bpw.weight.T

            cl_unbinned = ccl.cls.angular_cl(cosmo, tracer1, tracer2, ells_theory)

            cl_binned = np.dot(w_bins, cl_unbinned)

            cl_binned_list.append(cl_binned)


        cl_binned_total = np.concatenate(cl_binned_list)

        return cl_binned_total
