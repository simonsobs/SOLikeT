"""
.. module:: clusters

:Synopsis: Poisson likelihood for SZ clusters for Simons Osbervatory
:Authors: Nick Battaglia, Eunseong Lee

Likelihood for unbinned tSZ galaxy cluster number counts. Currently under development and
should be used only with caution and advice. Uses the SZ scaling relations from
Hasselfield et al (2013) [1]_ to compare observed number of :math:`y`-map detections
with the prediction from a Tinker [2]_ Halo Mass Function.

References
----------
.. [1] Hasselfield et al, JCAP 07, 008 (2013) `arXiv:1301.0816
                                                <https://arxiv.org/abs/1301.0816>`_
.. [2] Tinker et al, Astrophys. J. 688, 2, 709 (2008) `arXiv:0803.2706
                                                    <https://arxiv.org/abs/0803.2706>`_
p
"""
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from soliket.clusters import massfunc as mf
from soliket.poisson import PoissonLikelihood

from .survey import SurveyData
from .sz_utils import szutils, trapezoid
from cobaya import LoggedError

C_KM_S = 2.99792e5


class SZModel:
    pass


class ClusterLikelihood(PoissonLikelihood):
    """
    Poisson Likelihood for un-binned :math:`y`-map galaxy cluster counts.
    """

    name = "Clusters"
    columns = ["tsz_signal", "z", "tsz_signal_err"]

    # data_name = resource_filename("soliket",
    #                   "clusters/data/MFMF_WebSkyHalos_A10tSZ_3freq_tiles_mass.fits")

    def initialize(self):
        self.data_path = self.data_path or os.path.join(
            self.get_class_path(), "data", "selFn_equD56"
        )
        self.data_name = os.path.join(
            self.get_class_path(), "data", "E-D56Clusters.fits"
        )

        self.zarr = np.arange(0, 2, 0.05)
        self.k = np.logspace(-4, np.log10(5), 200)
        # self.mdef = self.ccl.halos.MassDef(500, 'critical')

        try:
            import pyccl as ccl
        except ImportError:
            raise LoggedError(
                self.log,
                "Could not import ccl. " "Install pyccl to use ClusterLikelihood.",
            )
        else:
            self.ccl = ccl
        super().initialize()

    def get_requirements(self):
        """
        This likelihood require :math:`P(k,z)`, :math:`H(z)`, :math:`d_A(z)`,
        :math:`r(z)` (co-moving radial distance) from Theory codes.

        :return: Dictionary of requirements
        """
        return {
            "Pk_interpolator": {
                "z": self.zarr,
                "k_max": 5.0,
                "nonlinear": False,
                "hubble_units": False,  # cobaya told me to
                "k_hunit": False,  # cobaya told me to
                "vars_pairs": [["delta_nonu", "delta_nonu"]],
            },
            "Hubble": {"z": self.zarr},
            "angular_diameter_distance": {"z": self.zarr},
            "comoving_radial_distance": {"z": self.zarr}
            # "CCL": {"methods": {"sz_model": self._get_sz_model}, "kmax": 10},
        }

    # def _get_sz_model(self, cosmo):
    #     model = SZModel()
    #     model.hmf = self.ccl.halos.MassFuncTinker08(cosmo, mass_def=self.mdef)
    #     model.hmb = self.ccl.halos.HaloBiasTinker10(
    #         cosmo, mass_def=self.mdef, mass_def_strict=False
    #     )
    #     model.hmc = self.ccl.halos.HMCalculator(cosmo, model.hmf, model.hmb, self.mdef)
    #     # model.szk = SZTracer(cosmo)
    #     return model

    def _get_catalog(self):
        self.survey = SurveyData(
            self.data_path, self.data_name
        )  # , MattMock=False,tiles=False)

        self.szutils = szutils(self.survey)

        df = pd.DataFrame(
            {
                "z": self.survey.clst_z.byteswap().newbyteorder(),
                "tsz_signal": self.survey.clst_y0.byteswap().newbyteorder(),
                "tsz_signal_err": self.survey.clst_y0err.byteswap().newbyteorder(),
            }
        )
        return df

    def _get_om(self):
        return (self.provider.get_param("omch2") + self.provider.get_param("ombh2")) / (
            (self.provider.get_param("H0") / 100.0) ** 2
        )

    def _get_ob(self):
        return (self.provider.get_param("ombh2")) / (
            (self.provider.get_param("H0") / 100.0) ** 2
        )

    def _get_Ez(self):
        return self.provider.get_Hubble(self.zarr) / self.provider.get_param("H0")

    def _get_Ez_interpolator(self):
        return interp1d(self.zarr, self._get_Ez())

    def _get_DAz(self):
        return self.provider.get_angular_diameter_distance(self.zarr)

    def _get_DAz_interpolator(self):
        return interp1d(self.zarr, self._get_DAz())

    def _get_HMF(self):
        h = self.provider.get_param("H0") / 100.0

        Pk_interpolator = self.provider.get_Pk_interpolator(
            ("delta_nonu", "delta_nonu"), nonlinear=False
        ).P
        pks = Pk_interpolator(self.zarr, self.k)
        # pkstest = Pk_interpolator(0.125, self.k )
        # print (pkstest * h**3 )

        Ez = (
            self._get_Ez()
        )  # self.provider.get_Hubble(self.zarr) / self.provider.get_param("H0")
        om = self._get_om()

        hmf = mf.HMF(om, Ez, pk=pks * h**3, kh=self.k / h, zarr=self.zarr)

        return hmf

    def _get_param_vals(self, **kwargs):
        # Read in scaling relation parameters
        # scat = kwargs['scat']
        # massbias = kwargs['massbias']
        # B0 = kwargs['B']
        B0 = 0.08
        scat = 0.2
        massbias = 1.0

        H0 = self.provider.get_param("H0")
        ob = self._get_ob()
        om = self._get_om()
        param_vals = {
            "om": om,
            "ob": ob,
            "H0": H0,
            "B0": B0,
            "scat": scat,
            "massbias": massbias,
        }
        return param_vals

    def _get_rate_fn(self, **kwargs):
        """
        Calculates the observed rate of clusters from the provided catalogue, which is
        then compared directly to the predicted rate at the current parameter values.
        """
        HMF = self._get_HMF()
        param_vals = self._get_param_vals(**kwargs)

        Ez_fn = self._get_Ez_interpolator()
        DA_fn = self._get_DAz_interpolator()

        dn_dzdm_interp = HMF.inter_dndmLogm(delta=500)

        h = self.provider.get_param("H0") / 100.0

        def Prob_per_cluster(z, tsz_signal, tsz_signal_err):
            c_y = tsz_signal
            c_yerr = tsz_signal_err
            c_z = z

            Pfunc_ind = self.szutils.Pfunc_per(
                HMF.M, c_z, c_y * 1e-4, c_yerr * 1e-4, param_vals, Ez_fn, DA_fn
            )

            dn_dzdm = (
                10 ** np.squeeze(dn_dzdm_interp((np.log10(HMF.M), c_z))) * h**4.0
            )

            ans = trapezoid(dn_dzdm * Pfunc_ind, dx=np.diff(HMF.M, axis=0), axis=0)
            return ans

        return Prob_per_cluster
        # Implement a function that returns a rate function (function of (tsz_signal, z))

    def _get_dVdz(self):
        DA_z = self.provider.get_angular_diameter_distance(self.zarr)

        dV_dz = (
            DA_z**2
            * (1.0 + self.zarr) ** 2
            / (self.provider.get_Hubble(self.zarr) / C_KM_S)
        )

        # dV_dz *= (self.provider.get_param("H0") / 100.0) ** 3.0  # was h0
        return dV_dz

    def _get_n_expected(self, **kwargs):
        """
        Calculates expected number of clusters at the current parameter values.
        """
        # def Ntot_survey(self,int_HMF,fsky,Ythresh,param_vals):

        HMF = self._get_HMF()
        param_vals = self._get_param_vals(**kwargs)
        Ez_fn = self._get_Ez_interpolator()
        DA_fn = self._get_DAz_interpolator()

        z_arr = self.zarr

        h = self.provider.get_param("H0") / 100.0

        Ntot = 0
        dVdz = self._get_dVdz()
        dn_dzdm = HMF.dn_dM(HMF.M, 500.0) * h**4.0  # getting rid of hs

        for Yt, frac in zip(self.survey.Ythresh, self.survey.frac_of_survey):
            Pfunc = self.szutils.PfuncY(Yt, HMF.M, z_arr, param_vals, Ez_fn, DA_fn)
            N_z = trapezoid(
                dn_dzdm * Pfunc, dx=np.diff(HMF.M[:, None] / h, axis=0), axis=0
            )
            Ntot += (
                trapezoid(N_z * dVdz, x=z_arr)
                * 4.0
                * np.pi
                * self.survey.fskytotal
                * frac
            )

        return Ntot

    def _test_n_tot(self, **kwargs):
        HMF = self._get_HMF()
        # param_vals = self._get_param_vals(**kwargs)
        # Ez_fn = self._get_Ez_interpolator()
        # DA_fn = self._get_DAz_interpolator()

        z_arr = self.zarr

        h = self.provider.get_param("H0") / 100.0

        Ntot = 0
        dVdz = self._get_dVdz()
        dn_dzdm = HMF.dn_dM(HMF.M, 500.0) * h**4.0  # getting rid of hs
        # Test Mass function against Nemo.
        Pfunc = 1.0
        N_z = trapezoid(dn_dzdm * Pfunc, dx=np.diff(HMF.M[:, None] / h, axis=0), axis=0)
        Ntot = (
            trapezoid(N_z * dVdz, x=z_arr)
            * 4.0
            * np.pi
            * (600.0 / (4 * np.pi * (180 / np.pi) ** 2))
        )

        return Ntot
