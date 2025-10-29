import numpy as np
from scipy import special
from scipy.interpolate import interp1d
from scipy.signal import convolve
from .cosmo import AngDist
from .gnfw import r200, rho_gnfw1h, Pth_gnfw1h, rho_gnfw, Pth_gnfw
from .obb import con, fstar_func, return_prof_pars, rho, Pth
from .beam import read_beam, f_beam, f_beam_fft, f_response
from .flat_map import FlatMap
from .hankel_transform_class import RadialFourierTransform
from ..constants import (
    MPC2CM,
    C_M_S,
    h_Planck,
    k_Boltzmann,
    electron_mass_kg,
    proton_mass_kg,
    hydrogen_fraction,
    T_CMB,
    ST_CGS,
)

kpc_cgs = MPC2CM * 1.0e-3
C_CGS = C_M_S * 1.0e2
ME_CGS = electron_mass_kg * 1.0e3
MP_CGS = proton_mass_kg * 1.0e3
sr2sqarcmin = 3282.8 * 60.0**2
XH = hydrogen_fraction


def radius_definition(transform_type):
    # There is probably a cleaner way to do this...
    if transform_type == "FFT":
        # New default from 15-500 (Amodeo21) to 30-1000 (Moser23)
        sizeArcmin = 30.0  # [degree]
        sz = 1000
        baseMap = FlatMap(
            nX=sz,
            nY=sz,
            sizeX=sizeArcmin * np.pi / 180.0 / 60.0,
            sizeY=sizeArcmin * np.pi / 180.0 / 60.0,
        )
        # map of the cluster, not yet convolved with the beam
        xc = baseMap.sizeX / 2.0
        yc = baseMap.sizeY / 2.0
        r = np.sqrt(
            (baseMap.x - xc) ** 2 + (baseMap.y - yc) ** 2
        )  # cluster centric radius in rad
        r_max = np.max(r)

    elif transform_type == "Hankel":
        baseMap = False
        r = False
        sizeArcmin = 30.0
        r_max = sizeArcmin * np.pi / 180.0 / 60.0 / np.sqrt(2)
    return baseMap, r, r_max


def coth(x):
    return 1 / np.tanh(x)


def fnu(nu):
    """input frequency in GHz"""
    nu *= 1e9
    x = h_Planck * nu / (k_Boltzmann * T_CMB)
    ans = x * coth(x / 2.0) - 4.0
    return ans


def convolve_FFT(
    r, thta_smooth, prof2D, beam_txt, baseMap, thta_use, beam_response=False
):
    profMap = np.interp(r, thta_smooth, prof2D)
    beamMapF = f_beam_fft(beam_txt, baseMap.ell)
    # Fourier transform the profile
    profMapF = baseMap.fourier(profMap)  # see note in flat_map.py about precision
    # multiply by the beam transfer function
    convolvedProfMapF = profMapF * beamMapF

    if beam_response is not False:  # for tSZ
        respTF = f_response(beam_response, baseMap.ell)
        # multiply by the response
        convolvedProfMapF *= respTF

    # inverse fourier transform
    convolvedProfMap = baseMap.inverseFourier(convolvedProfMapF)
    prof2D_beam = interp1d(
        r.flatten(),
        convolvedProfMap.flatten(),
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )(thta_use)
    return prof2D_beam


def convolve_Hankel(thta_smooth, prof2D, beam_txt, thta_use, beam_response=False):
    rht = RadialFourierTransform(
        n=200, pad=100, lrange=[170.0, 1.4e6]
    )  # note hard values, n here needs to be same size as rad in project
    profMap = np.interp(rht.r, thta_smooth, prof2D)
    lprofs = rht.real2harm(profMap)
    lprofs *= f_beam_fft(beam_txt, rht.ell)

    if beam_response is not False:
        respTF = f_response(beam_response, rht.ell)
        # multiply by the response
        lprofs *= respTF

    rprofs = rht.harm2real(lprofs)
    # padding
    r_unpad, rprofs = rht.unpad(rht.r, rprofs)
    prof2D_beam = interp1d(
        r_unpad.flatten(),
        rprofs.flatten(),
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )(thta_use)
    return prof2D_beam


def project_ksz(
    tht, M, z, beam_txt, transform_type, model_params, twohalo_term, provider
):  # input_model
    disc_fac = np.sqrt(2)
    NNR = 100
    resolution_factor = 2.0
    NNR2 = resolution_factor * NNR
    AngDis = AngDist(z, provider)

    baseMap, r, r_max = radius_definition(transform_type)

    r_use = AngDis * np.arctan(np.radians(tht / 60.0))
    r_use2 = AngDis * np.arctan(np.radians(tht * disc_fac / 60.0))
    r_ext = AngDis * np.arctan(r_max)  # total profile
    r_ext2 = r_ext

    rad = np.logspace(-3, 1, 200)  # Mpc
    rad2 = rad

    radlim = r_ext
    radlim2 = r_ext2

    dtht = np.arctan(radlim / AngDis) / NNR  # rads
    dtht2 = np.arctan(radlim2 / AngDis) / NNR  # rads
    dtht_use = np.arctan(r_use / AngDis) / NNR
    dtht2_use = np.arctan(r_use2 / AngDis) / NNR

    thta_use = (np.arange(NNR) + 1.0) * dtht_use
    thta2_use = (np.arange(NNR) + 1.0) * dtht2_use

    thta_smooth = (np.arange(NNR2) + 1.0) * dtht / resolution_factor
    thta2_smooth = (np.arange(NNR2) + 1.0) * dtht2 / resolution_factor
    thta_smooth = thta_smooth[:, None]
    thta2_smooth = thta2_smooth[:, None]

    rint = np.sqrt(rad**2 + thta_smooth**2 * AngDis**2)
    rint2 = np.sqrt(rad2**2 + thta2_smooth**2 * AngDis**2)

    # choose the model for projection
    # models = {
    #    "gnfw":rho_gnfw,
    #    "obb":rho
    # }
    # chosen_model = models.get(input_model)

    rho2D = (
        2
        * np.trapz(
            rho_gnfw(rint, M, z, model_params, twohalo_term, provider),
            x=rad * kpc_cgs,
            axis=1,
        )
        * 1e3
    )
    rho2D2 = (
        2
        * np.trapz(
            rho_gnfw(rint2, M, z, model_params, twohalo_term, provider),
            x=rad2 * kpc_cgs,
            axis=1,
        )
        * 1e3
    )

    thta_smooth = (np.arange(NNR2) + 1.0) * dtht / resolution_factor
    thta2_smooth = (np.arange(NNR2) + 1.0) * dtht2 / resolution_factor

    if transform_type == "FFT":
        rho2D_beam = convolve_FFT(r, thta_smooth, rho2D, beam_txt, baseMap, thta_use)
        rho2D2_beam = convolve_FFT(
            r, thta2_smooth, rho2D2, beam_txt, baseMap, thta2_use
        )

    elif transform_type == "Hankel":
        rho2D_beam = convolve_Hankel(thta_smooth, rho2D, beam_txt, thta_use)
        rho2D2_beam = convolve_Hankel(thta2_smooth, rho2D2, beam_txt, thta2_use)

    sig = 2.0 * np.pi * dtht_use * np.sum(thta_use * rho2D_beam)
    sig2 = 2.0 * np.pi * dtht2_use * np.sum(thta2_use * rho2D2_beam)

    sig_all_beam = (
        (2 * sig - sig2)
        * provider.get_param("v_rms")
        * ST_CGS
        * T_CMB
        * 1e6
        * ((1.0 + XH) / 2)
        / MP_CGS
    )  # units in muK*sr
    # sig_all_beam *= sr2sqarcmin #units in muK*sqarcmin
    return sig_all_beam


def project_tsz(
    tht,
    M,
    z,
    nu,
    beam_txt,
    transform_type,
    model_params,
    beam_response,
    twohalo_term,
    provider,
):
    disc_fac = np.sqrt(2)
    NNR = 100
    resolution_factor = 3.5
    NNR2 = resolution_factor * NNR
    AngDis = AngDist(z, provider)

    baseMap, r, r_max = radius_definition(transform_type)

    r_use = AngDis * np.arctan(np.radians(tht / 60.0))
    r_use2 = AngDis * np.arctan(np.radians(tht * disc_fac / 60.0))
    r_ext = AngDis * np.arctan(r_max)  # total profile
    r_ext2 = r_ext

    rad = np.logspace(-3, 1, 200)  # Mpc
    rad2 = rad

    radlim = r_ext
    radlim2 = r_ext2

    dtht = np.arctan(radlim / AngDis) / NNR  # rads
    dtht2 = np.arctan(radlim2 / AngDis) / NNR  # rads
    dtht_use = np.arctan(r_use / AngDis) / NNR
    dtht2_use = np.arctan(r_use2 / AngDis) / NNR

    thta_use = (np.arange(NNR) + 1.0) * dtht_use
    thta2_use = (np.arange(NNR) + 1.0) * dtht2_use

    thta_smooth = (np.arange(NNR2) + 1.0) * dtht / resolution_factor
    thta2_smooth = (np.arange(NNR2) + 1.0) * dtht2 / resolution_factor
    thta_smooth = thta_smooth[:, None]
    thta2_smooth = thta2_smooth[:, None]

    rint = np.sqrt(rad**2 + thta_smooth**2 * AngDis**2)
    rint2 = np.sqrt(rad2**2 + thta2_smooth**2 * AngDis**2)

    Pth2D = (
        2
        * np.trapz(
            Pth_gnfw(rint, M, z, model_params, twohalo_term, provider),
            x=rad * kpc_cgs,
            axis=1,
        )
        * 1e3
    )
    Pth2D2 = (
        2
        * np.trapz(
            Pth_gnfw(rint2, M, z, model_params, twohalo_term, provider),
            x=rad2 * kpc_cgs,
            axis=1,
        )
        * 1e3
    )

    thta_smooth = (np.arange(NNR2) + 1.0) * dtht / resolution_factor
    thta2_smooth = (np.arange(NNR2) + 1.0) * dtht2 / resolution_factor

    if transform_type == "FFT":
        Pth2D_beam = convolve_FFT(
            r, thta_smooth, Pth2D, beam_txt, baseMap, thta_use, beam_response
        )
        Pth2D2_beam = convolve_FFT(
            r, thta2_smooth, Pth2D2, beam_txt, baseMap, thta2_use, beam_response
        )
    elif transform_type == "Hankel":
        Pth2D_beam = convolve_Hankel(
            thta_smooth, Pth2D, beam_txt, thta_use, beam_response
        )
        Pth2D2_beam = convolve_Hankel(
            thta2_smooth, Pth2D2, beam_txt, thta2_use, beam_response
        )

    sig_p = 2.0 * np.pi * dtht_use * np.sum(thta_use * Pth2D_beam)
    sig2_p = 2.0 * np.pi * dtht2_use * np.sum(thta2_use * Pth2D2_beam)
    sig_all_p_beam = (
        (2 * sig_p - sig2_p)
        * ST_CGS
        / (ME_CGS * C_CGS**2)
        * ((2.0 + 2.0 * XH) / (3.0 + 5.0 * XH))
        * 1e6
    )

    # sig_all_p_beam *= sr2sqarcmin #units in muK*sqarcmin
    return sig_all_p_beam


def project_obb(tht, M, z, beam_txt, theta, nu, fbeam, provider):
    # NOTE: this is convolving through analytical integral, not FFT or FHT
    # NOTE: needs to be updated with new method for r from baseMap and Hankel transform
    disc_fac = np.sqrt(2)
    NNR = 100
    resolution_factor = 3.0
    NNR2 = resolution_factor * NNR

    theta_r = [theta[0], theta[1], theta[2], theta[3]]
    theta_p = [theta[0], theta[1], theta[2], theta[4]]

    AngDis = AngDist(z)

    r_ext = AngDis * np.arctan(np.radians(tht / 60.0))
    r_ext2 = AngDis * np.arctan(np.radians(tht * disc_fac / 60.0))

    rad = np.logspace(-3, 1, 200)
    rad2 = np.logspace(-3, 1, 200)

    radlim = r_ext
    radlim2 = r_ext2

    dtht = np.arctan(radlim / AngDis) / NNR
    dtht2 = np.arctan(radlim2 / AngDis) / NNR

    thta = (np.arange(NNR) + 1.0) * dtht
    thta2 = (np.arange(NNR) + 1.0) * dtht2

    thta_smooth = (np.arange(NNR2) + 1.0) * dtht / resolution_factor
    thta2_smooth = (np.arange(NNR2) + 1.0) * dtht2 / resolution_factor

    thta_smooth = thta_smooth[:, None]
    thta2_smooth = thta2_smooth[:, None]

    rint = np.sqrt(rad**2 + thta_smooth**2 * AngDis**2)
    rint2 = np.sqrt(rad2**2 + thta2_smooth**2 * AngDis**2)

    rho2D = 2 * np.trapz(rho(rint, M, z, theta_r), x=rad * kpc_cgs, axis=1) * 1e3
    rho2D2 = 2 * np.trapz(rho(rint2, M, z, theta_r), x=rad2 * kpc_cgs, axis=1) * 1e3
    Pth2D = 2 * np.trapz(Pth(rint, M, z, theta_p), x=rad * kpc_cgs, axis=1) * 1e3
    Pth2D2 = 2 * np.trapz(Pth(rint2, M, z, theta_p), x=rad2 * kpc_cgs, axis=1) * 1e3

    thta_smooth = (np.arange(NNR2) + 1.0) * dtht / resolution_factor
    thta = thta[:, None, None]
    thta2_smooth = (np.arange(NNR2) + 1.0) * dtht2 / resolution_factor
    thta2 = thta2[:, None, None]

    phi = np.linspace(0.0, 2 * np.pi, 100)
    phi = phi[None, None, :]
    thta_smooth = thta_smooth[None, :, None]
    thta2_smooth = thta2_smooth[None, :, None]

    rho2D = rho2D[None, :, None]
    rho2D2 = rho2D2[None, :, None]
    Pth2D = Pth2D[None, :, None]
    Pth2D2 = Pth2D2[None, :, None]

    b = read_beam(beam_txt)

    rho2D_beam0 = np.trapz(
        thta_smooth
        * rho2D
        * f_beam(
            np.sqrt(
                thta**2 + thta_smooth**2 - 2 * thta * thta_smooth * np.cos(phi)
            ),
            b,
        ),
        x=phi,
        axis=2,
    )

    rho2D2_beam0 = np.trapz(
        thta2_smooth
        * rho2D2
        * f_beam(
            np.sqrt(
                thta2**2 + thta2_smooth**2 - 2 * thta2 * thta2_smooth * np.cos(phi)
            ),
            b,
        ),
        x=phi,
        axis=2,
    )

    Pth2D_beam0 = np.trapz(
        thta_smooth
        * Pth2D
        * f_beam(
            np.sqrt(
                thta**2 + thta_smooth**2 - 2 * thta * thta_smooth * np.cos(phi)
            ),
            b,
        ),
        x=phi,
        axis=2,
    )

    Pth2D2_beam0 = np.trapz(
        thta2_smooth
        * Pth2D2
        * f_beam(
            np.sqrt(
                thta2**2 + thta2_smooth**2 - 2 * thta2 * thta2_smooth * np.cos(phi)
            ),
            b,
        ),
        x=phi,
        axis=2,
    )

    thta_smooth = (np.arange(NNR2) + 1.0) * dtht / resolution_factor
    thta2_smooth = (np.arange(NNR2) + 1.0) * dtht2 / resolution_factor

    rho2D_beam = np.trapz(rho2D_beam0, x=thta_smooth, axis=1)
    rho2D2_beam = np.trapz(rho2D2_beam0, x=thta2_smooth, axis=1)
    Pth2D_beam = np.trapz(Pth2D_beam0, x=thta_smooth, axis=1)
    Pth2D2_beam = np.trapz(Pth2D2_beam0, x=thta2_smooth, axis=1)

    thta = (np.arange(NNR) + 1.0) * dtht
    thta2 = (np.arange(NNR) + 1.0) * dtht2

    sig = 2.0 * np.pi * dtht * np.sum(thta * rho2D_beam)
    sig2 = 2.0 * np.pi * dtht2 * np.sum(thta2 * rho2D2_beam)

    sig_all_beam = (
        (2 * sig - sig2)
        * provider.get_param("v_rms")
        * ST_CGS
        * T_CMB
        * 1e6
        * ((1.0 + XH) / 2.0)
        / MP_CGS
    )

    sig_p = 2.0 * np.pi * dtht * np.sum(thta * Pth2D_beam)
    sig2_p = 2.0 * np.pi * dtht2 * np.sum(thta2 * Pth2D2_beam)

    sig_all_p_beam = (
        fnu(nu)
        * (2 * sig_p - sig2_p)
        * ST_CGS
        / (ME_CGS * C_CGS**2)
        * T_CMB
        * 1e6
        * ((2.0 + 2.0 * XH) / (3.0 + 5.0 * XH))
    )
    # sig_all_beam *= sr2sqarcmin
    # sig_all_p_beam *= sr2sqarcmin
    return sig_all_beam, sig_all_p_beam
