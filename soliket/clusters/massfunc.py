import numpy as np
from scipy.interpolate import interp2d
from .tinker import dn_dlogM

np.seterr(divide='ignore', invalid='ignore')

MSUN_CGS = 1.98840987e+33
G_CGS = 6.67259e-08
MPC2CM = 3.085678e+24


class HMF:
    def __init__(self, om, Ez, pk=None, kh=None, zarr=None):

        # Initialize redshift and mass ranges
        if zarr is None:
            self.zarr = np.arange(0.05, 1.95, 0.1)
        else:
            self.zarr = zarr

        # self.M = 10**np.arange(np.log10(5e13), 15.7, 0.02)
        # self.M = 10**np.arange(13.5, 15.7, 0.02)
        M_edges = 10 ** np.arange(13.5, 15.72, 0.02)

        self.M = (M_edges[1:] + M_edges[:-1]) / 2.  # 10**np.arange(13.5, 15.7, 0.02)

        assert len(Ez) == len(zarr), "Ez and z arrays do not match"

        self.E_z = Ez

        # Initialize rho critical values for usage
        self.om = om
        self.rho_crit0H100 = (3. / (8. * np.pi) * (100 * 1.e5) ** 2.) \
                                / G_CGS * MPC2CM / MSUN_CGS
        self.rhoc0om = self.rho_crit0H100 * self.om

        if pk is None:
            print('this will not work')
        else:
            self.pk = pk
            self.kh = kh
            # self.kh, self.pk = self._pk(self.zarr)

    def rhoc(self):
        # critical density as a function of z
        ans = self.rho_crit0H100 * self.E_z ** 2.
        return ans

    def rhom(self):
        # mean matter density as a function of z
        ans = self.rhoc0om * (1.0 + self.zarr) ** 3
        return ans

    def critdensThreshold(self, deltac):
        rho_treshold = deltac * self.rhoc() / self.rhom()
        return rho_treshold

    def dn_dM(self, M, delta):
        """
        dN/dmdV Mass Function
        M here is in MDeltam but we can convert
        """
        delts = self.critdensThreshold(delta)
        dn_dlnm = dn_dlogM(M, self.zarr, self.rhoc0om, delts, self.kh, self.pk,
                           'comoving')
        dn_dm = dn_dlnm / M[:, None]
        return dn_dm

    def inter_dndmLogm(self, delta, M=None):
        """
        interpolating over M and z for faster calculations
        """
        if M is None:
            M = self.M
        dndM = self.dn_dM(M, delta)
        ans = interp2d(self.zarr, np.log10(M), np.log10(dndM), kind='cubic', fill_value=0)
        return ans
