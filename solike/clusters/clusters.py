"""
requires extra: astlib

"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pkg_resources import resource_filename

from ..poisson import PoissonLikelihood
from . import massfunc as mf
from .survey import SurveyData
from .sz_utils import szutils

C_KM_S = 2.99792e5


class ClusterLikelihood(PoissonLikelihood):
    name = "Clusters"
    columns = ["tsz_signal", "z", "tsz_signal_err"]
    data_path = resource_filename("solike", "clusters/data/selFn_equD56")
    data_name = resource_filename("solike", "clusters/data/ACTPol_Cond_scatv5.fits")

    def initialize(self):
        self.zarr = np.arange(0, 2, 0.05)
        self.k = np.logspace(-4, np.log10(5), 200)
        super().initialize()

    def get_requirements(self):
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
        }

    def _get_catalog(self):
        self.survey = SurveyData(self.data_path, self.data_name, szarMock=True)

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
        return (self.theory.get_param("omch2") + self.theory.get_param("ombh2")) / (
            (self.theory.get_param("H0") / 100.0) ** 2
        )

    def _get_ob(self):
        return (self.theory.get_param("ombh2")) / ((self.theory.get_param("H0") / 100.0) ** 2)

    def _get_Ez(self):
        return self.theory.get_Hubble(self.zarr) / self.theory.get_param("H0")

    def _get_Ez_interpolator(self):
        return interp1d(self.zarr, self._get_Ez())

    def _get_DAz(self):
        return self.theory.get_angular_diameter_distance(self.zarr)

    def _get_DAz_interpolator(self):
        return interp1d(self.zarr, self._get_DAz())

    def _get_HMF(self):
        h = self.theory.get_param("H0") / 100.0

        Pk_interpolator = self.theory.get_Pk_interpolator(("delta_nonu", "delta_nonu"), nonlinear=False).P
        pks = Pk_interpolator(self.zarr, self.k)
        # pkstest = Pk_interpolator(0.125, self.k )
        # print (pkstest * h**3 )

        Ez = self._get_Ez()  # self.theory.get_Hubble(self.zarr) / self.theory.get_param("H0")
        om = self._get_om()

        hmf = mf.HMF(om, Ez, pk=pks * h ** 3, kh=self.k / h, zarr=self.zarr)

        return hmf

    def _get_param_vals(self):
        B0 = 0.08
        scat = 0.2
        massbias = 1.0
        H0 = self.theory.get_param("H0")
        ob = self._get_ob()
        om = self._get_om()
        param_vals = {"om": om, "ob": ob, "H0": H0, "B0": B0, "scat": scat, "massbias": massbias}
        return param_vals

    def _get_rate_fn(self, **kwargs):
        HMF = self._get_HMF()
        param_vals = self._get_param_vals()

        Ez_fn = self._get_Ez_interpolator()
        DA_fn = self._get_DAz_interpolator()

        dn_dzdm_interp = HMF.inter_dndmLogm(delta=500)

        h = self.theory.get_param("H0") / 100.0

        def Prob_per_cluster(z, tsz_signal, tsz_signal_err):
            c_y = tsz_signal
            c_yerr = tsz_signal_err
            c_z = z

            Pfunc_ind = self.szutils.Pfunc_per(
                HMF.M, c_z, c_y * 1e-4, c_yerr * 1e-4, param_vals, Ez_fn, DA_fn
            )

            dn_dzdm = 10 ** np.squeeze(dn_dzdm_interp(c_z, np.log10(HMF.M))) * h ** 4.0

            ans = np.trapz(dn_dzdm * Pfunc_ind, dx=np.diff(HMF.M, axis=0), axis=0)
            # import pdb

            # pdb.set_trace()
            return ans

        return Prob_per_cluster
        # Implement a function that returns a rate function (function of (tsz_signal, z))

    def _get_dVdz(self):
        """dV/dzdOmega
        """
        DA_z = self.theory.get_angular_diameter_distance(self.zarr)

        dV_dz = DA_z ** 2 * (1.0 + self.zarr) ** 2 / (self.theory.get_Hubble(self.zarr) / C_KM_S)

        # dV_dz *= (self.theory.get_param("H0") / 100.0) ** 3.0  # was h0
        return dV_dz

    def _get_n_expected(self, **kwargs):
        # def Ntot_survey(self,int_HMF,fsky,Ythresh,param_vals):

        HMF = self._get_HMF()
        param_vals = self._get_param_vals()
        Ez_fn = self._get_Ez_interpolator()
        DA_fn = self._get_DAz_interpolator()

        z_arr = self.zarr

        h = self.theory.get_param("H0") / 100.0

        Ntot = 0
        dVdz = self._get_dVdz()
        dn_dzdm = HMF.dn_dM(HMF.M, 500.0) * h ** 4.0  # getting rid of hs

        for Yt, frac in zip(self.survey.Ythresh, self.survey.frac_of_survey):
            Pfunc = self.szutils.PfuncY(Yt, HMF.M, z_arr, param_vals, Ez_fn, DA_fn)
            N_z = np.trapz(dn_dzdm * Pfunc, dx=np.diff(HMF.M[:, None] / h, axis=0), axis=0)
            Ntot += np.trapz(N_z * dVdz, x=z_arr) * 4.0 * np.pi * self.survey.fskytotal * frac

        # To test Mass function against Nemo.
        # Pfunc = 1.
        # N_z = np.trapz(dn_dzdm * Pfunc, dx=np.diff(HMF.M[:, None]/h, axis=0), axis=0)
        # Ntot = np.trapz(N_z * dVdz, x=z_arr) * 4.0 * np.pi * (600./(4*np.pi * (180/np.pi)**2))
        print("Ntot", Ntot)

        return Ntot
