import numpy as np
from scipy import special
from scipy.signal import convolve
from .cosmo import AngDist
from .gnfw import r200, rho_gnfw1h, Pth_gnfw1h, rho_gnfw, Pth_gnfw
from .obb import con, fstar_func, return_prof_pars, rho, Pth
from .beam import read_beam, f_beam

from ..constants import MPC2CM, C_M_S, h_Planck, k_Boltzmann, electron_mass_kg, proton_mass_kg, hydrogen_fraction, T_CMB, ST_CGS


fb = cosmo_params["Omega_b"] / cosmo_params["Omega_m"] #put cosmo params in the yaml file, look at cross corr for example, param_values['cobaya_name']
kpc_cgs = MPC2CM * 1.e-3
C_CGS = C_M_S * 1.e2
ME_CGS = electron_mass_kg * 1.e3
MP_CGS = proton_mass_kg * 1.e3
sr2sqarcmin = 3282.8 * 60.**2
XH = hydrogen_fraction

def coth(x):
    return 1 / np.tanh(x)


def fnu(nu):
    """input frequency in GHz"""
    nu *= 1e9
    x = h_Planck * nu / (k_Boltzmann * T_CMB)
    ans = x * coth(x / 2.0) - 4.0
    return ans


def project_ksz(tht, M, z, beam_txt, gnfw_params):
    disc_fac = np.sqrt(2)
    l0 = 30000.0
    NNR = 100
    NNR2 = 2.0 * NNR

    drint = 1e-3 * (kpc_cgs * 1e3)
    AngDis = AngDist(z)

    r_ext = AngDis * np.arctan(np.radians(tht / 60.0))
    r_ext2 = AngDis * np.arctan(np.radians(tht * disc_fac / 60.0))

    rad = np.logspace(-3, 1, 200)  # Mpc
    rad2 = np.logspace(-3, 1, 200)

    radlim = r_ext
    radlim2 = r_ext2

    dtht = np.arctan(radlim / AngDis) / NNR  # rads
    dtht2 = np.arctan(radlim2 / AngDis) / NNR  # rads

    thta = (np.arange(NNR) + 1.0) * dtht
    thta2 = (np.arange(NNR) + 1.0) * dtht2

    thta_smooth = (np.arange(NNR2) + 1.0) * dtht
    thta2_smooth = (np.arange(NNR2) + 1.0) * dtht2

    thta_smooth = thta_smooth[:, None]
    thta2_smooth = thta2_smooth[:, None]

    rint = np.sqrt(rad ** 2 + thta_smooth ** 2 * AngDis ** 2)
    rint2 = np.sqrt(rad2 ** 2 + thta2_smooth ** 2 * AngDis ** 2)

    rho2D = 2 * np.trapz(rho_gnfw(rint, M, z, gnfw_params), x=rad * kpc_cgs, axis=1) * 1e3
    rho2D2 = 2 * np.trapz(rho_gnfw(rint2, M, z, gnfw_params), x=rad2 * kpc_cgs, axis=1) * 1e3

    thta_smooth = (np.arange(NNR2) + 1.0) * dtht
    thta = thta[:, None, None]
    thta2_smooth = (np.arange(NNR2) + 1.0) * dtht2
    thta2 = thta2[:, None, None]

    phi = np.linspace(0.0, 2 * np.pi, 100)
    phi = phi[None, None, :]
    thta_smooth = thta_smooth[None, :, None]
    thta2_smooth = thta2_smooth[None, :, None]

    rho2D = rho2D[None, :, None]
    rho2D2 = rho2D2[None, :, None]

    b = read_beam(beam_txt)

    rho2D_beam0 = np.trapz(
        thta_smooth
        * rho2D
        * f_beam(np.sqrt(thta ** 2 + thta_smooth ** 2 - 2 * thta * thta_smooth * np.cos(phi)),b),
        x=phi,
        axis=2,
    )

    rho2D2_beam0 = np.trapz(
        thta2_smooth
        * rho2D2
        * f_beam(np.sqrt(thta2 ** 2 + thta2_smooth ** 2 - 2 * thta2 * thta2_smooth * np.cos(phi)),b),
        x=phi,
        axis=2,
    )

    thta_smooth = (np.arange(NNR2) + 1.0) * dtht
    thta2_smooth = (np.arange(NNR2) + 1.0) * dtht2

    rho2D_beam = np.trapz(rho2D_beam0, x=thta_smooth, axis=1)
    rho2D2_beam = np.trapz(rho2D2_beam0, x=thta2_smooth, axis=1)

    thta = (np.arange(NNR) + 1.0) * dtht
    thta2 = (np.arange(NNR) + 1.0) * dtht2

    area_fac = 2.0 * np.pi * dtht * np.sum(thta)

    sig = 2.0 * np.pi * dtht * np.sum(thta * rho2D_beam)
    sig2 = 2.0 * np.pi * dtht2 * np.sum(thta2 * rho2D2_beam)

    sig_all_beam = (
        (2 * sig - sig2) * v_rms * ST_CGS * TCMB * 1e6 * ((2.0 + 2.0 * XH) / (3.0 + 5.0 * XH)) / MP_CGS
    ) #units in muK*sr
    sig_all_beam *= sr2sqarcmin #units in muK*sqarcmin
    return sig_all_beam

def project_tsz(tht, M, z, nu, fbeam, gnfw_params):
    disc_fac = np.sqrt(2)
    l0 = 30000.0
    NNR = 100
    NNR2 = 3.5 * NNR
    
    drint = 1e-3 * (kpc_cgs * 1e3)
    AngDis = AngDist(z)

    rvir = r200(M,z) / kpc_cgs / 1e3  # in MPC

 	r_ext = AngDis * np.arctan(np.radians(tht / 60.0))
    r_ext2 = AngDis * np.arctan(np.radians(tht * disc_fac / 60.0))

    rad = np.logspace(-3, 1, 200)  # Mpc
    rad2 = np.logspace(-3, 1, 200)

    radlim = r_ext
    radlim2 = r_ext2

    dtht = np.arctan(radlim / AngDis) / NNR  # rads
    dtht2 = np.arctan(radlim2 / AngDis) / NNR  # rads

    thta = (np.arange(NNR) + 1.0) * dtht
    thta2 = (np.arange(NNR) + 1.0) * dtht2

    thta_smooth = (np.arange(NNR2) + 1.0) * dtht
    thta2_smooth = (np.arange(NNR2) + 1.0) * dtht2

    thta_smooth = thta_smooth[:, None]
    thta2_smooth = thta2_smooth[:, None]

    rint = np.sqrt(rad ** 2 + thta_smooth ** 2 * AngDis ** 2)
    rint2 = np.sqrt(rad2 ** 2 + thta2_smooth ** 2 * AngDis ** 2)

    Pth2D = 2 * np.trapz(Pth_gnfw(rint, M, z, gnfw_params), x=rad * kpc_cgs, axis=1) * 1e3
    Pth2D2 = 2 * np.trapz(Pth_gnfw(rint2, M, z, gnfw_params), x=rad2 * kpc_cgs, axis=1) * 1e3

    thta_smooth = (np.arange(NNR2) + 1.0) * dtht
    thta = thta[:, None, None]
    thta2_smooth = (np.arange(NNR2) + 1.0) * dtht2
    thta2 = thta2[:, None, None]
    phi = np.linspace(0.0, 2 * np.pi, 50)
    phi = phi[None, None, :]
    thta_smooth = thta_smooth[None, :, None]
    thta2_smooth = thta2_smooth[None, :, None]
    Pth2D = Pth2D[None, :, None]
    Pth2D2 = Pth2D2[None, :, None]

    b = read_beam(beam_txt)

    Pth2D_beam0 = np.trapz(
        thta_smooth
        * Pth2D
        * f_beam(np.sqrt(thta ** 2 + thta_smooth ** 2 - 2 * thta * thta_smooth * np.cos(phi)),b),
        x=phi,
        axis=2,
    )

    Pth2D2_beam0 = np.trapz(
        thta2_smooth
        * Pth2D2
        * f_beam(np.sqrt(thta2 ** 2 + thta2_smooth ** 2 - 2 * thta2 * thta2_smooth * np.cos(phi)),b),
        x=phi,
        axis=2,
    )

    thta_smooth = (np.arange(NNR2) + 1.0) * dtht
    thta2_smooth = (np.arange(NNR2) + 1.0) * dtht2

    Pth2D_beam = np.trapz(Pth2D_beam0, x=thta_smooth, axis=1)
    Pth2D2_beam = np.trapz(Pth2D2_beam0, x=thta2_smooth, axis=1)

    thta = (np.arange(NNR) + 1.0) * dtht
    thta2 = (np.arange(NNR) + 1.0) * dtht2

    area_fac = 2.0 * np.pi * dtht * np.sum(thta)

    sig_p = 2.0 * np.pi * dtht * np.sum(thta * Pth2D_beam)
    sig2_p = 2.0 * np.pi * dtht2 * np.sum(thta2 * Pth2D2_beam)

    sig_all_p_beam = (
        fnu(nu)
        * (2 * sig_p - sig2_p)
        * ST_CGS
        / (ME_CGS * C_CGS ** 2)
        * TCMB
        * 1e6
        * ((2.0 + 2.0 * XH) / (3.0 + 5.0 * XH))
    )  #units in muK*sr
    sig_all_p_beam *= sr2sqarcmin #units in muK*sqarcmin
    return sig_all_p_beam