"""
requires extra: astlib

"""
import numpy as np
import pandas as pd

from solike.poisson import PoissonLikelihood
import solike.clusters.massfunc as mf

from .survey import SurveyData


class ClusterLikelihood(PoissonLikelihood):
    class_options = {
        "name": "Clusters",
        "columns": ["tsz_signal", "z"],
        "data_path": "selFn_equD56",
        "data_name": "ACTPol_Cond_scatv5.fits",
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
        catalog = SurveyData(self.data_path, self.data_name, szarMock=True)
        self.catalog = catalog

        df = pd.DataFrame(
            {
                "z": self.catalog.clst_z.byteswap().newbyteorder(),
                "tsz_signal": self.catalog.clst_y0.byteswap().newbyteorder(),
                "tsz_signal_err": self.catalog.clst_y0err.byteswap().newbyteorder(),
            }
        )
        return df

    def _get_HMF(self):

        Pk_interpolator = self.theory.get_Pk_interpolator(("delta_nonu", "delta_nonu"), nonlinear=False).P
        pks = Pk_interpolator(self.zarr, self.k)
        Ez = self.theory.get_Hubble(self.zarr) / self.theory.get_param("H0")
        om = (self.theory.get_param("omch2") + self.theory.get_param("ombh2")) / (
            (self.theory.get_param("H0") / 100.0) ** 2
        )

        hmf = mf.HMF(om, Ez, pk=pks, kh=self.k, zarr=self.zarr)

        return hmf

    def _get_rate_fn(self, **kwargs):

        HMF = self._get_HMF()
        param_vals = ...

        def Prob_per_cluster(z, tsz_signal, tsz_signal_err):
            y = tsz_signal
            y_err = tsz_signal_err

            tempz = cluster_props[0, :]
            zind = np.argsort(tempz)
            tempz = 0.0
            c_z = cluster_props[0, zind]
            c_zerr = cluster_props[1, zind]
            c_y = cluster_props[2, zind]
            c_yerr = cluster_props[3, zind]

            Marr = np.outer(int_HMF.M.copy(), np.ones([len(c_z)]))
            zarr = np.outer(np.ones([len(int_HMF.M.copy())]), c_z)

            if c_zerr.any() > 0:
                # FIX THIS
                z_arr = np.arange(-3.0 * c_zerr, (3.0 + 0.1) * c_zerr, c_zerr) + c_z
                Pfunc_ind = self.Pfunc_per_zarr(int_HMF.M.copy(), z_arr, c_y, c_yerr, int_HMF, param_vals)
                M200 = int_HMF.cc.Mass_con_del_2_del_mean200(int_HMF.M.copy(), 500, c_z)  # FIX THIS?
                dn_dzdm = dn_dzdm_int(z_arr, np.log10(int_HMF.M.copy()))
                N_z_ind = np.trapz(dn_dzdm * Pfunc_ind, dx=np.diff(M200, axis=0), axis=0)
                N_per = np.trapz(N_z_ind * gaussian(z_arr, c_z, c_zerr), dx=np.diff(z_arr))
                ans = N_per
            else:
                Pfunc_ind = self.Pfunc_per(Marr, zarr, c_y, c_yerr, param_vals)
                dn_dzdm = HMF.dn_dzdm(c_z, np.log10(int_HMF.M.copy()))
                M200 = int_HMF.M200_int(c_z, int_HMF.M.copy())
                N_z_ind = np.trapz(dn_dzdm * Pfunc_ind, dx=np.diff(M200, axis=0), axis=0)
                ans = N_z_ind

            ans = ...

            return ans

        return Prob_per_cluster
        # Implement a function that returns a rate function (function of (tsz_signal, z))

    def _get_n_expected(self, **kwargs):
        pass
        # Implement integral of the above.
