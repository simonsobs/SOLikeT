"""
.. module:: massfunction

The ``HMF`` class build the halo mass function internally required for the cluster
likelihood. Calculates the Halo Mass Function as in Tinker et al (2008) [2]_ .

"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from soliket.constants import G_CGS, MPC2CM, MSUN_CGS

from .tinker import dn_dlogM

np.seterr(divide='ignore', invalid='ignore')


class HMF:
    """
    Build halo mass function
    """
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

    def rhoc(self) -> np.ndarray:
        """
        Critical density as a function of z
        """
        return self.rho_crit0H100 * self.E_z ** 2.

    def rhom(self) -> np.ndarray:
        """
        Mean matter density as a function of z
        """
        return self.rhoc0om * (1.0 + self.zarr) ** 3

    def critdensThreshold(self, deltac) -> np.ndarray:
        return deltac * self.rhoc() / self.rhom()

    def dn_dM(self, M, delta) -> np.ndarray:
        """
        dN/dmdV Mass Function

        :param M: Mass in MDeltam, but we can convert
        :param delta: Threshold for critical density
        """
        delts = self.critdensThreshold(delta)
        dn_dlnm = dn_dlogM(M, self.zarr, self.rhoc0om, delts, self.kh, self.pk,
                           'comoving')
        dn_dm = dn_dlnm / M[:, None]
        return dn_dm

    def inter_dndmLogm(self, delta, M=None) -> RegularGridInterpolator:
        """
        Interpolating over M and z for faster calculations
        """
        if M is None:
            M = self.M
        dndM = self.dn_dM(M, delta)
        return RegularGridInterpolator((np.log10(M), self.zarr),
                                       np.log10(dndM), method='cubic', fill_value=0)
