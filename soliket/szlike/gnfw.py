import numpy as np
from .cosmo import rho_cz

from ..constants import MSUN_CGS, G_CGS, MPC2CM

kpc_cgs = MPC2CM * 1.0e-3


def r200(M, z, provider):
    """radius of a sphere with density 200 times the critical density of the universe.
    Input mass in solar masses. Output radius in cm.
    """
    M_cgs = M * MSUN_CGS
    om = provider.get_param("Omega_m")
    ol = provider.get_param("Omega_L")
    Ez2 = om * (1 + z) ** 3 + ol
    rhocrit = 1.87847e-29 * provider.get_param("hh") ** 2
    ans = (3 * M_cgs / (4 * np.pi * 200.0 * rhocrit * Ez2)) ** (1.0 / 3.0)
    return ans


def rho_gnfw1h(x, M, z, theta, provider):
    # Bins and weights specific to CMASS
    b_cen = np.array([[12.27689266, 12.67884686, 13.16053855, 13.69871423]]).T
    p = np.array([4.13431979e-03, 1.31666601e-01, 3.36540698e-01, 8.13760167e-02])
    rho = []

    fb = provider.get_param("Omega_b") / provider.get_param("Omega_m")
    for i in range(0, len(b_cen)):
        m = 10 ** b_cen[i]
        r200c = r200(m, z, provider)
        rvir = r200c / kpc_cgs / 1e3  # Mpc
        xc = 0.5
        al = 0.88 * (m / 1e14) ** (-0.03) * (1 + z) ** 0.19
        gm = -0.2
        rho0, bt = theta
        rho.append(
            10**rho0
            * (x / rvir / xc) ** gm
            / ((1 + (x / rvir / xc) ** al) ** ((bt - gm) / al))
            * rho_cz(z, provider)
            * fb
        )
    rho = np.array(rho)
    rho_av = np.average(rho, weights=p, axis=0)
    return rho_av


def rho_gnfw2h(xx, theta2h, twohalo_term):
    # Average 2h specific to CMASS, make option for importing file
    # or using two_halo script to calculate
    rho_file = np.genfromtxt(twohalo_term)
    x1 = rho_file[:, 0]
    rho2h = rho_file[:, 1]
    ans = np.interp(xx, x1, rho2h)
    return theta2h * ans


def rho_gnfw(xx, M, z, theta, twohalo_term, provider):
    theta1h = theta[0], theta[1]
    theta2h = theta[2]
    ans = rho_gnfw1h(xx, M, z, theta1h, provider) + rho_gnfw2h(
        xx, theta2h, twohalo_term
    )
    return ans


def Pth_gnfw1h(x, M, z, theta, provider):
    # Bins and weights specific to CMASS
    """
    b_cen = np.array(
        [
            [
                11.31932504,
                11.43785913,
                11.57526319,
                11.74539764,
                11.97016907,
                12.27689266,
                12.67884686,
                13.16053855,
                13.69871423,
            ]
        ]
    ).T
    p = np.array(
        [
            2.94467222e-06,
            2.94467222e-06,
            2.94467222e-06,
            1.47233611e-05,
            3.38637305e-05,
            4.13431979e-03,
            1.31666601e-01,
            3.36540698e-01,
            8.13760167e-02,
        ]
    )
    """
    b_cen = np.array([[12.27689266, 12.67884686, 13.16053855, 13.69871423]]).T
    p = np.array([4.13431979e-03, 1.31666601e-01, 3.36540698e-01, 8.13760167e-02])

    pth = []
    fb = provider.get_param("Omega_b") / provider.get_param("Omega_m")
    for i in range(0, len(b_cen)):
        m = 10 ** b_cen[i]
        r200c = r200(m, z, provider)
        rvir = r200c / kpc_cgs / 1e3  # Mpc
        M_cgs = m * MSUN_CGS
        P200c = G_CGS * M_cgs * 200.0 * rho_cz(z, provider) * fb / (2.0 * r200c)
        P0, bt = theta
        al = 1.0
        xc = 0.497 * (m / 1e14) ** (-0.00865) * (1 + z) ** 0.731
        gm = -0.3
        pth.append(
            P0 * (x / rvir / xc) ** gm * (1 + (x / rvir / xc) ** al) ** (-bt) * P200c
        )
    pth = np.array(pth)
    pth_av = np.average(pth, weights=p, axis=0)
    return pth_av


def Pth_gnfw2h(xx, theta2h, twohalo_term):
    # 2h specific to CMASS, make option for importing file
    # or using two_halo script to calculate
    pth_file = np.genfromtxt(twohalo_term)
    x1 = pth_file[:, 0]
    pth2h = pth_file[:, 2]
    ans = np.interp(xx, x1, pth2h)
    return theta2h * ans


def Pth_gnfw(xx, M, z, theta, twohalo_term, provider):
    theta1h = theta[0], theta[1]
    theta2h = theta[2]
    ans = Pth_gnfw1h(xx, M, z, theta1h, provider) + Pth_gnfw2h(
        xx, theta2h, twohalo_term
    )
    return ans
