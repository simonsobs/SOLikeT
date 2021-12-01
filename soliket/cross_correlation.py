"""
Simple likelihood for CMB-galaxy cross-correlations using the cobaya
CCL module.

First version by Pablo Lemos
"""

import numpy as np
from .gaussian import GaussianData, GaussianLikelihood
import pyccl as ccl


class CrossCorrelationLikelihood(GaussianLikelihood):
    def initialize(self):
        self.dndz = np.loadtxt(self.dndz_file)

        x, y, dy = self._get_data()
        cov = np.diag(dy**2)
        self.data = GaussianData("CrossCorrelation", x, y, cov)

    def get_requirements(self):
        return {'CCL': {"kmax": 10,
                        "nonlinear": True}}

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
        cosmo = self.provider.get_CCL()['cosmo']

        tracer_g = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=self.dndz.T,
                                          bias=(self.dndz[:, 0],
                                                params_values['b1'] *
                                                np.ones(len(self.dndz[:, 0]))),
                                          mag_bias=(self.dndz[:, 0],
                                                    params_values['s1'] *
                                                    np.ones(len(self.dndz[:, 0])))
                                          )
        tracer_k = ccl.CMBLensingTracer(cosmo, z_source=1060)

        cl_gg = ccl.cls.angular_cl(cosmo, tracer_g, tracer_g, self.ell_auto)  # + 1e-7
        cl_kg = ccl.cls.angular_cl(cosmo, tracer_k, tracer_g, self.ell_cross)

        return np.concatenate([cl_gg, cl_kg])


class ShearKappaLikelihood(CrossCorrelationLikelihood):

    def _get_theory(self, **params_values):
        cosmo = self.provider.get_CCL()['cosmo']

        # import pdb
        # pdb.set_trace()

        tracer_gamma = ccl.WeakLensingTracer(cosmo, dndz=self.dndz.T,
                                             ia_bias=(self.dndz[:, 0],
                                                      params_values['A_IA'] *
                                                      np.ones(len(self.dndz[:, 0])))
                                             )
        tracer_k = ccl.CMBLensingTracer(cosmo, z_source=1060)

        cl_gammagamma = ccl.cls.angular_cl(cosmo,
                                           tracer_gamma, tracer_gamma,
                                           self.ell_auto)
        cl_kgamma = ccl.cls.angular_cl(cosmo,
                                       tracer_k, tracer_gamma,
                                       self.ell_cross)

        return np.concatenate([cl_gammagamma, cl_kgamma])
