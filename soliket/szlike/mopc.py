import numpy as np
from scipy import special
from scipy.signal import convolve
from .params import cosmo_params
from .cosmo import AngDist
from .gnfw import r200, rho_gnfw1h, Pth_gnfw1h, rho_gnfw, Pth_gnfw
from .obb import con, fstar_func, return_prof_pars, rho, Pth  # , rho1h_one_mass, Pth1h_one_mass

# from .batt import rho_batt_mw, Pth_batt_mw
import matplotlib.pyplot as plt
import time

from tqdm import tqdm


"""
The OBB model of rho and Pth is obtained through 'make_a_obs_profile()'.

The GNFW model of rho is obtained through 'make_a_obs_profile_sim_rho()'.

The GNFW model of Pth is obtained through 'make_a_obs_profile_sim_pth()'.
"""

fb = cosmo_params["Omega_b"] / cosmo_params["Omega_m"]
kBoltzmann = 1.380658e-23  # m2 kg /(s2 K)
hPlanck = 6.6260755e-34  # m2 kg/s
Gmax = 0.216216538797
Gravity = 6.67259e-8
rhocrit = 1.87847e-29 * cosmo_params["hh"] ** 2
Msol_cgs = 1.989e33
kpc_cgs = 3.086e21
kev_erg = 1.60218e-9
C_CGS = 2.99792e10
ST_CGS = 6.65246e-25
ME_CGS = 9.10939e-28
MP_CGS = 1.6726219e-24
TCMB = 2.725
GMAX = 0.2162
XH = 0.76
delx = 0.01
v_rms = 1.06e-3  # 1e-3 #v_rms/c


def coth(x):
    return 1 / np.tanh(x)


def fnu(nu):
    """input frequency in GHz"""
    nu *= 1e9
    x = hPlanck * nu / (kBoltzmann * TCMB)
    ans = x * coth(x / 2.0) - 4.0
    return ans


# def project_prof_beam(tht,M,z,theta,theta2,nu,f_beam):
def project_prof_beam(tht, M, z, theta, nu, fbeam):
    disc_fac = np.sqrt(2)
    l0 = 30000.0
    NNR = 100
    NNR2 = 3 * NNR

    # fwhm = beam
    # fwhm *= np.pi / (180.*60.)
    # sigmaBeam = fwhm / np.sqrt(8.*np.log(2.))

    # P0, rho0, x_f = theta2
    # fstar = fstar_func(M)

    AngDis = AngDist(z)

    rvir = r200(M, z) / kpc_cgs / 1e3  # in MPC
    c = con(M, z)

    r_ext = AngDis * np.arctan(np.radians(tht / 60.0))
    r_ext2 = AngDis * np.arctan(np.radians(tht * disc_fac / 60.0))

    rvir_arcmin = 180.0 * 60.0 / np.pi * np.tan(rvir / AngDis)  # arcmin
    rvir_ext = AngDis * np.arctan(np.radians(rvir_arcmin / 60.0))
    rvir_ext2 = AngDis * np.arctan(np.radians(rvir_arcmin * disc_fac / 60.0))

    rad = np.logspace(-3, 1, 2e2)  # in MPC
    rad2 = np.logspace(-3, 1, 2e2)  # in MPC

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

    rho2D = 2 * np.trapz(rho(rint, M, z, theta), x=rad * kpc_cgs, axis=1) * 1e3
    rho2D2 = 2 * np.trapz(rho(rint2, M, z, theta), x=rad2 * kpc_cgs, axis=1) * 1e3
    Pth2D = 2 * np.trapz(Pth(rint, M, z, theta), x=rad * kpc_cgs, axis=1) * 1e3
    Pth2D2 = 2 * np.trapz(Pth(rint2, M, z, theta), x=rad2 * kpc_cgs, axis=1) * 1e3

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
    Pth2D = Pth2D[None, :, None]
    Pth2D2 = Pth2D2[None, :, None]

    rho2D_beam0 = np.trapz(
        thta_smooth
        * rho2D
        * f_beam(np.sqrt(thta ** 2 + thta_smooth ** 2 - 2 * thta * thta_smooth * np.cos(phi))),
        x=phi,
        axis=2,
    )

    rho2D2_beam0 = np.trapz(
        thta2_smooth
        * rho2D2
        * f_beam(np.sqrt(thta2 ** 2 + thta2_smooth ** 2 - 2 * thta2 * thta2_smooth * np.cos(phi))),
        x=phi,
        axis=2,
    )

    Pth2D_beam0 = np.trapz(
        thta_smooth
        * Pth2D
        * f_beam(np.sqrt(thta ** 2 + thta_smooth ** 2 - 2 * thta * thta_smooth * np.cos(phi))),
        x=phi,
        axis=2,
    )

    Pth2D2_beam0 = np.trapz(
        thta2_smooth
        * Pth2D2
        * f_beam(np.sqrt(thta2 ** 2 + thta2_smooth ** 2 - 2 * thta2 * thta2_smooth * np.cos(phi))),
        x=phi,
        axis=2,
    )

    thta_smooth = (np.arange(NNR2) + 1.0) * dtht
    thta2_smooth = (np.arange(NNR2) + 1.0) * dtht2

    rho2D_beam = np.trapz(rho2D_beam0, x=thta_smooth, axis=1)
    rho2D2_beam = np.trapz(rho2D2_beam0, x=thta2_smooth, axis=1)
    Pth2D_beam = np.trapz(Pth2D_beam0, x=thta_smooth, axis=1)
    Pth2D2_beam = np.trapz(Pth2D2_beam0, x=thta2_smooth, axis=1)

    thta = (np.arange(NNR) + 1.0) * dtht
    thta2 = (np.arange(NNR) + 1.0) * dtht2

    area_fac = 2.0 * np.pi * dtht * np.sum(thta)

    sig = 2.0 * np.pi * dtht * np.sum(thta * rho2D_beam)
    sig2 = 2.0 * np.pi * dtht2 * np.sum(thta2 * rho2D2_beam)

    sig_all_beam = (
        (2 * sig - sig2) * v_rms * ST_CGS * TCMB * 1e6 * ((2.0 + 2.0 * XH) / (3.0 + 5.0 * XH)) / MP_CGS
    )

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
    )  # / area_fac # muK
    return sig_all_beam, sig_all_p_beam


def project_prof_beam_sim_rho(tht, M, z, theta_rho, f_beam):
    theta_sim_rho = theta_rho

    disc_fac = np.sqrt(2)
    l0 = 30000.0
    NNR = 100
    NNR2 = 2.0 * NNR

    # fwhm = beam #arcmin
    # fwhm *= np.pi / (180.*60.) #rad
    # sigmaBeam = fwhm / np.sqrt(8.*np.log(2.))

    drint = 1e-3 * (kpc_cgs * 1e3)
    XH = 0.76

    AngDis = AngDist(z)

    # rvir = r200(M,z)/kpc_cgs/1e3 #Mpc
    # c = con(M,z)

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

    rho2D = 2 * np.trapz(rho_gnfw(rint, M, z, theta_sim_rho), x=rad * kpc_cgs, axis=1) * 1e3
    rho2D2 = 2 * np.trapz(rho_gnfw(rint2, M, z, theta_sim_rho), x=rad2 * kpc_cgs, axis=1) * 1e3

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

    rho2D_beam0 = np.trapz(
        thta_smooth
        * rho2D
        * f_beam(np.sqrt(thta ** 2 + thta_smooth ** 2 - 2 * thta * thta_smooth * np.cos(phi))),
        x=phi,
        axis=2,
    )

    rho2D2_beam0 = np.trapz(
        thta2_smooth
        * rho2D2
        * f_beam(np.sqrt(thta2 ** 2 + thta2_smooth ** 2 - 2 * thta2 * thta2_smooth * np.cos(phi))),
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
    )  # / (np.pi * np.radians(tht/60.)**2)

    return sig_all_beam


def project_prof_beam_sim_pth(tht, M, z, theta_pth, nu, f_beam):
    theta_sim_pth = theta_pth

    disc_fac = np.sqrt(2)
    l0 = 30000.0
    NNR = 100
    NNR2 = 3.5 * NNR

    # fwhm = beam
    # fwhm *= np.pi / (180.*60.)
    # sigmaBeam = fwhm / np.sqrt(8.*np.log(2.))

    drint = 1e-3 * (kpc_cgs * 1e3)
    XH = 0.76

    AngDis = AngDist(z)

    m_med = np.median(M)
    # rvir = r200(m_med,z)/kpc_cgs/1e3 #Mpc

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

    Pth2D = 2 * np.trapz(Pth_gnfw(rint, M, z, theta_sim_pth), x=rad * kpc_cgs, axis=1) * 1e3
    Pth2D2 = 2 * np.trapz(Pth_gnfw(rint2, M, z, theta_sim_pth), x=rad2 * kpc_cgs, axis=1) * 1e3

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

    Pth2D_beam0 = np.trapz(
        thta_smooth
        * Pth2D
        * f_beam(np.sqrt(thta ** 2 + thta_smooth ** 2 - 2 * thta * thta_smooth * np.cos(phi))),
        x=phi,
        axis=2,
    )

    Pth2D2_beam0 = np.trapz(
        thta2_smooth
        * Pth2D2
        * f_beam(np.sqrt(thta2 ** 2 + thta2_smooth ** 2 - 2 * thta2 * thta2_smooth * np.cos(phi))),
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
    )  # / area_fac # muK

    return sig_all_p_beam


def project_prof_beam_sim_y(tht, M, z, theta_pth, beam):
    theta_sim_pth = theta_pth

    disc_fac = np.sqrt(2)
    l0 = 30000.0
    NNR = 100
    NNR2 = 3.5 * NNR

    fwhm = beam
    fwhm *= np.pi / (180.0 * 60.0)
    sigmaBeam = fwhm / np.sqrt(8.0 * np.log(2.0))

    drint = 1e-3 * (kpc_cgs * 1e3)
    XH = 0.76

    AngDis = AngDist(z)

    m_med = np.median(M)
    # rvir = r200(m_med,z)/kpc_cgs/1e3 #Mpc

    r_ext = AngDis * np.arctan(np.radians(tht / 60.0))
    r_ext2 = AngDis * np.arctan(np.radians(tht * disc_fac / 60.0))

    rad = np.logspace(-3, 1, 2e2)  # Mpc
    rad2 = np.logspace(-3, 1, 2e2)

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

    Pth2D = 2 * np.trapz(Pth_gnfw(rint, M, z, theta_sim_pth), x=rad * kpc_cgs, axis=1) * 1e3
    Pth2D2 = 2 * np.trapz(Pth_gnfw(rint2, M, z, theta_sim_pth), x=rad2 * kpc_cgs, axis=1) * 1e3

    thta_smooth = (np.arange(NNR2) + 1.0) * dtht
    thta = thta[:, None]
    thta2_smooth = (np.arange(NNR2) + 1.0) * dtht2
    thta2 = thta2[:, None]

    Pth2D_beam = np.trapz(
        thta_smooth
        * Pth2D
        * np.exp(-0.5 * thta_smooth ** 2 / sigmaBeam ** 2)
        * special.iv(0, thta_smooth * thta / sigmaBeam ** 2),
        x=thta_smooth,
        axis=1,
    )
    Pth2D2_beam = np.trapz(
        thta2_smooth
        * Pth2D2
        * np.exp(-0.5 * thta2_smooth ** 2 / sigmaBeam ** 2)
        * special.iv(0, thta2_smooth * thta2 / sigmaBeam ** 2),
        x=thta2_smooth,
        axis=1,
    )

    thta = (np.arange(NNR) + 1.0) * dtht
    thta2 = (np.arange(NNR) + 1.0) * dtht2

    area_fac = 2.0 * np.pi * dtht * np.sum(thta)

    Pth2D_beam *= np.exp(-0.5 * thta ** 2 / sigmaBeam ** 2) / sigmaBeam ** 2
    Pth2D2_beam *= np.exp(-0.5 * thta2 ** 2 / sigmaBeam ** 2) / sigmaBeam ** 2

    sig_p = 2.0 * np.pi * dtht * np.sum(thta * Pth2D_beam)
    sig2_p = 2.0 * np.pi * dtht2 * np.sum(thta2 * Pth2D2_beam)

    sig_all_p_beam = (
        (2 * sig_p - sig2_p) * ST_CGS / (ME_CGS * C_CGS ** 2) * ((2.0 + 2.0 * XH) / (3.0 + 5.0 * XH))
    )  # / area_fac # muK

    return sig_all_p_beam


def find_params_M(M, z, theta_0):
    theta0 = np.append([M, z], [theta_0])
    beta_0 = 1.1
    con_test = con(M, z)
    theta2 = np.array([beta_0, con_test * 1.01])
    ans = return_prof_pars(theta2, theta0)
    return ans


def make_a_obs_profile(thta_arc, M, z, theta_0, nu, fbeam):
    # thta2 = find_params_M(M,z,theta_0)
    rho = np.zeros(len(thta_arc))
    pth = np.zeros(len(thta_arc))
    for ii in range(len(thta_arc)):
        temp = project_prof_beam(thta_arc[ii], M, z, theta_0, nu, fbeam)
        rho[ii] = temp[0]
        pth[ii] = temp[1]
    return rho, pth


def make_a_obs_profile_sim_rho(thta_arc, M, z, theta_rho, f_beam):
    rho = np.zeros(len(thta_arc))
    for ii in tqdm(range(len(thta_arc))):
        temp = project_prof_beam_sim_rho(thta_arc[ii], M, z, theta_rho, f_beam)
        rho[ii] = temp
    return rho


def make_a_obs_profile_sim_pth(thta_arc, M, z, theta_pth, nu, f_beam):

    pth = np.zeros(len(thta_arc))
    for ii in tqdm(range(len(thta_arc))):
        temp = project_prof_beam_sim_pth(thta_arc[ii], M, z, theta_pth, nu, f_beam)
        pth[ii] = temp
    return pth


def make_a_obs_profile_sim_y(thta_arc, M, z, theta_pth, beam):
    pth = np.zeros(len(thta_arc))
    for ii in tqdm(range(len(thta_arc))):
        temp = project_prof_beam_sim_y(thta_arc[ii], M, z, theta_pth, beam)
        pth[ii] = temp
    return pth


def project_prof_sim_rho(rs, M, z, theta_rho):
    theta_sim_rho = theta_rho

    # rs in MPC
    NNR = 100
    rad = np.logspace(-3, 1, 2e2)  # Mpc

    rint = np.sqrt(rad[None, :] ** 2 + rs[:, None] ** 2)

    rhoint = rho_gnfw(rint, M, z, theta_sim_rho)
    rho2D = 2 * np.trapz(rhoint, x=rad * kpc_cgs, axis=1) * 1e3

    return rho2D  # Not normalized
