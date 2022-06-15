import numpy as np
from scipy import special
from scipy.interpolate import interp1d
from scipy.signal import convolve
from .cosmo import AngDist
from .gnfw import r200, rho_gnfw1h, Pth_gnfw1h, rho_gnfw, Pth_gnfw
from .obb import con, fstar_func, return_prof_pars, rho, Pth
from .beam import read_beam, f_beam
from .flat_map import FlatMap

from ..constants import MPC2CM, C_M_S, h_Planck, k_Boltzmann, electron_mass_kg, proton_mass_kg, hydrogen_fraction, T_CMB, ST_CGS

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


def project_ksz(tht, M, z, beam_txt, model_params, provider): #input_model
    disc_fac = np.sqrt(2)
    l0 = 30000.0
    NNR = 100
    NNR2 = 2.0 * NNR

    drint = 1e-3 * (kpc_cgs * 1e3)
    AngDis = AngDist(z, provider)

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

    #choose the model for projection
    #models = {
    #    "gnfw":rho_gnfw,
    #    "obb":rho
    #}
    #chosen_model = models.get(input_model)

    rho2D = 2 * np.trapz(rho_gnfw(rint, M, z, model_params, provider), x=rad * kpc_cgs, axis=1) * 1e3
    rho2D2 = 2 * np.trapz(rho_gnfw(rint2, M, z, model_params, provider), x=rad2 * kpc_cgs, axis=1) * 1e3

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
        (2 * sig - sig2) * provider.get_param('v_rms') * ST_CGS * T_CMB * 1e6 * ((2.0 + 2.0 * XH) / (3.0 + 5.0 * XH)) / MP_CGS
    ) #units in muK*sr
    sig_all_beam *= sr2sqarcmin #units in muK*sqarcmin
    return sig_all_beam

def project_tsz(tht, M, z, nu, beam_txt, model_params, beam_response, provider):
    disc_fac = np.sqrt(2)
    l0 = 30000.0
    NNR = 100
    NNR2 = 3.5 * NNR
    
    drint = 1e-3 * (kpc_cgs * 1e3)
    AngDis = AngDist(z, provider)

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

    Pth2D = 2 * np.trapz(Pth_gnfw(rint, M, z, model_params, provider), x=rad * kpc_cgs, axis=1) * 1e3
    Pth2D2 = 2 * np.trapz(Pth_gnfw(rint2, M, z, model_params, provider), x=rad2 * kpc_cgs, axis=1) * 1e3

    if beam_response is not False:
        sizeArcmin = 15.  # [degree]
        sz = 500 
        baseMap = FlatMap(nX=sz, nY=sz, sizeX=sizeArcmin*np.pi/180./60., sizeY=sizeArcmin*np.pi/180./60.)
        # map of the cluster, not yet convolved with the beam
        xc = baseMap.sizeX / 2.
        yc = baseMap.sizeY / 2.
        r = np.sqrt((baseMap.x-xc)**2 + (baseMap.y-yc)**2)  # cluster centric radius in rad

        thta_smooth = (np.arange(NNR2) + 1.)*dtht
        thta2_smooth = (np.arange(NNR2) + 1.)*dtht2

        profMap = np.interp(r,thta_smooth,Pth2D)
        profMap2 = np.interp(r,thta2_smooth,Pth2D2)
        
        ##figure out better ways to do this, but just matching the old format for now
        ell, beam = np.loadtxt(beam_txt,usecols = (0,1), unpack = True)
        beamMapF = np.interp(baseMap.l,ell,beam)
        ell, resp = np.loadtxt(beam_response, usecols = (0,1), unpack = True) 
        respTF = np.interp(baseMap.l,ell,resp) 

        # Fourier transform the profile
        profMapF = baseMap.fourier(profMap)
        profMapF2 = baseMap.fourier(profMap2)
        # multiply by the beam transfer function
        convolvedProfMapF = profMapF * beamMapF * respTF #test #test
        convolvedProfMapF2 = profMapF2 * beamMapF * respTF
        # inverse Fourier transform
        convolvedProfMap = baseMap.inverseFourier(convolvedProfMapF)
        convolvedProfMap2 = baseMap.inverseFourier(convolvedProfMapF2)

        thta = (np.arange(NNR) + 1.)*dtht
        thta2 = (np.arange(NNR) + 1.)*dtht2

        Pth2D_beam = interp1d(r.flatten(),convolvedProfMap.flatten(), kind='linear', bounds_error=False, fill_value=0.)(thta)
        Pth2D2_beam = interp1d(r.flatten(),convolvedProfMap2.flatten(), kind='linear', bounds_error=False, fill_value=0.)(thta2)

        area_fac = 2.0*np.pi*dtht*np.sum(thta)
        sig_p  = 2.0*np.pi*dtht*np.sum(thta*Pth2D_beam)
        sig2_p = 2.0*np.pi*dtht2*np.sum(thta2*Pth2D2_beam)
        sig_all_p_beam = (2*sig_p - sig2_p) * ST_CGS/(ME_CGS*C_CGS**2) * ((2. + 2.*XH)/(3.+5.*XH)) * 1e6

    else:
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
            * T_CMB
            * 1e6
            * ((2.0 + 2.0 * XH) / (3.0 + 5.0 * XH))
        )  #units in muK*sr
    sig_all_p_beam *= sr2sqarcmin #units in muK*sqarcmin
    return sig_all_p_beam

def project_obb(tht, M, z, theta, theta2, nu, fbeam): #this originally didn't have theta2??
    disc_fac = np.sqrt(2)
    l0 = 30000.0
    NNR = 100
    NNR2 = 3 * NNR

    AngDis = AngDist(z)

    rvir = r200(M, z) / kpc_cgs / 1e3
    c = con(M, z)

    r_ext = AngDis * np.arctan(np.radians(tht / 60.0))
    r_ext2 = AngDis * np.arctan(np.radians(tht * disc_fac / 60.0))

    rvir_arcmin = 180.0 * 60.0 / np.pi * np.tan(rvir / AngDis)
    rvir_ext = AngDis * np.arctan(np.radians(rvir_arcmin / 60.0))
    rvir_ext2 = AngDis * np.arctan(np.radians(rvir_arcmin * disc_fac / 60.0))

    rad = np.logspace(-3, 1, 200) 
    rad2 = np.logspace(-3, 1, 200)

    radlim = r_ext
    radlim2 = r_ext2

    dtht = np.arctan(radlim / AngDis) / NNR 
    dtht2 = np.arctan(radlim2 / AngDis) / NNR

    thta = (np.arange(NNR) + 1.0) * dtht
    thta2 = (np.arange(NNR) + 1.0) * dtht2

    thta_smooth = (np.arange(NNR2) + 1.0) * dtht
    thta2_smooth = (np.arange(NNR2) + 1.0) * dtht2

    thta_smooth = thta_smooth[:, None]
    thta2_smooth = thta2_smooth[:, None]

    rint = np.sqrt(rad ** 2 + thta_smooth ** 2 * AngDis ** 2)
    rint2 = np.sqrt(rad2 ** 2 + thta2_smooth ** 2 * AngDis ** 2)

    rho2D = 2 * np.trapz(rho(rint, M, z, theta, theta2), x=rad * kpc_cgs, axis=1) * 1e3 #theta2??
    rho2D2 = 2 * np.trapz(rho(rint2, M, z, theta, theta2), x=rad2 * kpc_cgs, axis=1) * 1e3
    Pth2D = 2 * np.trapz(Pth(rint, M, z, theta, theta2), x=rad * kpc_cgs, axis=1) * 1e3
    Pth2D2 = 2 * np.trapz(Pth(rint2, M, z, theta, theta2), x=rad2 * kpc_cgs, axis=1) * 1e3

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
    ) 
    sig_all_beam *= sr2sqarcmin
    sig_all_p_beam *= sr2sqarcmin
    return sig_all_beam, sig_all_p_beam