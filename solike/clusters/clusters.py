"""
requires extra: astlib

"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pkg_resources import resource_filename

from solike.poisson import PoissonLikelihood
import solike.clusters.massfunc as mf

from .survey import SurveyData
from .sz_utils import szutils


class ClusterLikelihood(PoissonLikelihood):
    class_options = {
        "name": "Clusters",
        "columns": ["tsz_signal", "z", "tsz_signal_err"],
        "data_path": resource_filename("solike.clusters", "data/selFn_equD56"),
        "data_name": resource_filename("solike.clusters", "data/ACTPol_Cond_scatv5.fits"),
        # "params": {""},
    }

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

    def _get_HMF(self):

        Pk_interpolator = self.theory.get_Pk_interpolator(("delta_nonu", "delta_nonu"), nonlinear=False).P
        pks = Pk_interpolator(self.zarr, self.k)
        Ez = self.theory.get_Hubble(self.zarr) / self.theory.get_param("H0")
        om = self._get_om()

        hmf = mf.HMF(om, Ez, pk=pks, kh=self.k, zarr=self.zarr)

        return hmf

    def _get_rate_fn(self, **kwargs):

        HMF = self._get_HMF()

        B0 = 0.08
        scat = 0.2
        massbias = 1.0
        H0 = self.theory.get_param("H0")
        ob = self._get_ob()
        om = self._get_om()
        param_vals = {"om": om, "ob": ob, "H0": H0, "B0": B0, "scat": scat, "massbias": massbias}

        Ez_fn = self._get_Ez_interpolator()

        def Prob_per_cluster(z, tsz_signal, tsz_signal_err):
            c_y = tsz_signal
            c_yerr = tsz_signal_err
            c_z = z

            Pfunc_ind = self.szutils.Pfunc_per(
                HMF.M[:, None], self.zarr[None, :], c_y, c_yerr, param_vals, Ez_fn
            )
            dn_dzdm = HMF.dn_dzdm(c_z, np.log10(HMF.M))

            ans = np.trapz(dn_dzdm * Pfunc_ind, dx=np.diff(HMF.M, axis=0), axis=0)

            return ans

        return Prob_per_cluster
        # Implement a function that returns a rate function (function of (tsz_signal, z))

    def _get_n_expected(self, **kwargs):
        pass
        # Implement integral of the above.
