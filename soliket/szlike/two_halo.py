import numpy as np
from scipy.integrate import quad
from .cosmo import rho_cz
from .gnfw import r200
from hmf import MassFunction, transfer
from colossus.lss import bias, peaks
from colossus.cosmology import cosmology


"""set cosmology"""
params = {
    "flat": True,
    "H0": 70.0,
    "Om0": 0.25,
    "Ob0": 0.044,
    "sigma8": 0.8159,
    "ns": 0.97,
}

"""parameters used in Illustris TNG https://arxiv.org/pdf/1703.02970.pdf"""
# params = {
#    'flat': True,
#    'H0': 67.7,
#    'Om0': 0.31,
#    'Ob0': 0.0486,
#    'sigma8': 0.8159,
#    'ns': 0.97,
# }

cosmology.setCosmology("myCosmo", params)

hh = 0.7
fb = params["Ob0"] / params["Om0"]
G_cgs = 6.67259e-8  # cm3/g/s2
rhocrit = 1.87847e-29 * hh**2
Msol_cgs = 1.989e33
kpc_cgs = 3.086e21

#################################################################################
# Computing the 2-halo component of density and preassure profiles in cgs units #
#################################################################################

# From Battaglia 2016, Appendix A
def rho_gnfw(x, m, z):
    rho0 = 4e3 * (m / 1e14) ** 0.29 * (1 + z) ** (-0.66)
    al = 0.88 * (m / 1e14) ** (-0.03) * (1 + z) ** 0.19
    bt = 3.83 * (m / 1e14) ** 0.04 * (1 + z) ** (-0.025)
    xc = 0.5
    gm = -0.2
    ans = rho0 * (x / xc) ** gm * (1 + (x / xc) ** al) ** (-(bt - gm) / al)
    ans *= rho_cz(z) * fb
    return ans


def rhoFourier(k, m, z):
    ans = []
    for i in range(len(m)):
        r200c = r200(m[i], z) / kpc_cgs / 1e3
        integrand = (
            lambda r: 4.0
            * np.pi
            * r**2
            * rho_gnfw(r / r200c, m[i], z)
            * np.sin(k * r)
            / (k * r)
        )
        res = quad(integrand, 0.0, 10 * r200c, epsabs=0.0, epsrel=1.0e-4, limit=10000)[
            0
        ]
        ans.append(res)
    ans = np.array(ans)
    return ans


def hmf(m, z):
    """Shet, Mo &  Tormen 2001"""
    Mmin = np.log10(np.min(m))
    Mmax = np.log10(np.max(m))
    return MassFunction(
        z=z, Mmin=Mmin, Mmax=Mmax, dlog10m=(Mmax - Mmin) / 49.5, hmf_model="SMT"
    ).dndm


def b(m, z):
    """Shet, Mo &  Tormen 2001"""
    nu = peaks.peakHeight(m, z)
    delta_c = peaks.collapseOverdensity(corrections=True, z=z)
    aa, bb, cc = 0.707, 0.5, 0.6
    return 1.0 + 1.0 / (np.sqrt(aa) * delta_c) * (
        np.sqrt(aa) * aa * nu**2
        + np.sqrt(aa) * bb * (aa * nu**2) ** (1 - cc)
        - ((aa * nu**2) ** cc / (aa * nu**2) ** cc + bb * (1 - cc) * (1 - cc / 2))
    )


def Plin(k, z):
    lnk_min = np.log(np.min(k))
    lnk_max = np.log(np.max(k))
    dlnk = (lnk_max - lnk_min) / (49.5)
    """Eisenstein & Hu (1998)"""
    p = transfer.Transfer(
        sigma_8=0.8344,
        n=0.9624,
        z=z,
        lnk_min=lnk_min,
        lnk_max=lnk_max,
        dlnk=dlnk,
        transfer_model="EH",
    )
    return p.power


def rho_2h(r, m, z):

    # first compute P_2h (power spectrum)
    m_array = np.logspace(np.log10(1.0e10), np.log10(1.0e15), 50, 10.0)
    k_array = np.logspace(np.log10(1.0e-3), np.log10(1.0e3), 50, 10.0)
    hmf_array = np.array([hmf(m_array, z)] * len(k_array)).reshape(
        len(k_array), len(hmf(m_array, z))
    )
    bias_array = np.array([b(m_array, z)] * len(k_array)).reshape(
        len(k_array), len(b(m_array, z))
    )

    arr = []
    for i in range(len(k_array)):
        arr.append(
            np.trapz(
                hmf_array[i, :] * bias_array[i, :] * rhoFourier(k_array[i], m_array, z),
                m_array,
            )
        )
    arr = np.array(arr)
    P2h = np.array(arr * b(m, z) * Plin(k_array, z))

    # then Fourier transform to get rho_2h
    rcorr = 50.0  # Mpc/h
    integrand = (
        lambda k: 1.0
        / (2 * np.pi**2.0)
        * k**2
        * np.sin(k * r)
        / (k * r)
        * np.interp(k, k_array, P2h)
        if k > 1.0 / rcorr
        else 0.0
    )
    res = quad(integrand, 0.0, np.inf, epsabs=0.0, epsrel=1.0e-2, limit=1000)[0]
    return res


# From Battaglia 2012, AGN Feedback Delta=200
def Pth_gnfw(x, m, z):
    P200c = G_cgs * m * Msol_cgs * 200.0 * rho_cz(z) * fb / (2.0 * r200(m, z))
    P0 = 18.1 * (m / 1e14) ** 0.154 * (1 + z) ** (-0.758)
    al = 1.0
    bt = 4.35 * (m / 1e14) ** 0.0393 * (1 + z) ** 0.415
    xc = 0.497 * (m / 1e14) ** (-0.00865) * (1 + z) ** 0.731
    gm = -0.3
    ans = P0 * (x / xc) ** gm * (1 + (x / xc) ** al) ** (-bt)
    ans *= P200c
    return ans


def PthFourier(k, m, z):
    ans = []
    for i in range(len(m)):
        r200c = r200(m[i], z) / kpc_cgs / 1e3
        integrand = (
            lambda r: 4.0
            * np.pi
            * r**2
            * Pth_gnfw(r / r200c, m[i], z)
            * np.sin(k * r)
            / (k * r)
        )
        res = quad(integrand, 0.0, 10 * r200c, epsabs=0.0, epsrel=1.0e-4, limit=10000)[
            0
        ]
        ans.append(res)
    ans = np.array(ans)
    return ans


def Pth_2h(r, m, z):
    # first compute P_2h (power spectrum)
    m_array = np.logspace(np.log10(1.0e10), np.log10(1.0e15), 50, 10.0)
    k_array = np.logspace(np.log10(1.0e-3), np.log10(1.0e3), 50, 10.0)
    hmf_array = np.array([hmf(m_array, z)] * len(k_array)).reshape(
        len(k_array), len(hmf(m_array, z))
    )
    bias_array = np.array([b(m_array, z)] * len(k_array)).reshape(
        len(k_array), len(b(m_array, z))
    )

    arr = []
    for i in range(len(k_array)):
        arr.append(
            np.trapz(
                hmf_array[i, :] * bias_array[i, :] * PthFourier(k_array[i], m_array, z),
                m_array,
            )
        )
    arr = np.array(arr)
    P2h = np.array(arr * b(m, z) * Plin(k_array, z))

    # then Fourier transform to get Pth_2h
    rcorr = 50.0  # Mpc/h
    integrand = (
        lambda k: 1.0
        / (2 * np.pi**2.0)
        * k**2
        * np.sin(k * r)
        / (k * r)
        * np.interp(k, k_array, P2h)
        if k > 1.0 / rcorr
        else 0.0
    )
    res = quad(integrand, 0.0, np.inf, epsabs=0.0, epsrel=1.0e-2, limit=1000)[0]
    return res
