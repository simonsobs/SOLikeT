'''
Likelihood for SZ model
'''

#original imports
import numpy as np
from scipy import special
from scipy.signal import convolve
from .params import cosmo_params
from .cosmo import AngDist
from .gnfw import r200, rho_gnfw1h, Pth_gnfw1h, rho_gnfw, Pth_gnfw
from .obb import con, fstar_func, return_prof_pars, rho, Pth

import matplotlib.pyplot as plt
import time

from tqdm import tqdm

#new imports
from .gaussian import GaussianData, GaussianLikelihood
from ..constants import h_Planck, k_Boltzmann, C_M_S, ST_CGS, electron_mass_kg, proton_mass_kg, T_CMB

fb = cosmo_params["Omega_b"] / cosmo_params["Omega_m"] #put cosmo params in the yaml file, look at cross corr for example, param_values['cobaya_name']
kpc_cgs = 3.086e21
C_CGS = C_M_S*1.e2
ME_CGS = electron_mass_kg * 1.e3
MP_CGS = proton_mass_kg * 1.e3
XH = 0.76 #can add things to the constants file but name them better
v_rms = 1.06e-3  # 1e-3 #v_rms/c


def coth(x):
    return 1 / np.tanh(x)


def fnu(nu):
    """input frequency in GHz"""
    nu *= 1e9
    x = h_Planck * nu / (k_Boltzmann * T_CMB)
    ans = x * coth(x / 2.0) - 4.0
    return ans

def project_prof_beam(tht, M, z, params_rho, params_pth, nu, fbeam):
    disc_fac = np.sqrt(2)
    l0 = 30000.0
    NNR = 100
    NNR2 = 3 * NNR

    AngDis = AngDist(z)

    rvir = r200(M, z) / kpc_cgs / 1e3  # in MPC

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

    rho2D = 2 * np.trapz(rho(rint, M, z, params_rho), x=rad * kpc_cgs, axis=1) * 1e3
    rho2D2 = 2 * np.trapz(rho(rint2, M, z, params_rho), x=rad2 * kpc_cgs, axis=1) * 1e3
    Pth2D = 2 * np.trapz(Pth(rint, M, z, params_pth), x=rad * kpc_cgs, axis=1) * 1e3
    Pth2D2 = 2 * np.trapz(Pth(rint2, M, z, params_pth), x=rad2 * kpc_cgs, axis=1) * 1e3

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
        (2 * sig - sig2) * v_rms * ST_CGS * T_CMB * 1e6 * ((2.0 + 2.0 * XH) / (3.0 + 5.0 * XH)) / MP_CGS
    )

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
    )
    return sig_all_beam, sig_all_p_beam


def make_a_obs_profile(thta_arc, M, z, theta_0, nu, fbeam): 
	#original version only had one set of theta for both rho and pth
	#but that doesn't make sense since we have different params?
    rho = np.zeros(len(thta_arc))
    pth = np.zeros(len(thta_arc))
    for ii in range(len(thta_arc)):
        temp = project_prof_beam(thta_arc[ii], M, z, theta_0, nu, fbeam)
        rho[ii] = temp[0]
        pth[ii] = temp[1]
    return rho, pth


class SZLikelihood(GaussianLikelihood):
    def initialize(self):
        #initialize beam file?
        self.beam=np.loadtxt(self.beam_file)
        #initialize z,nu,M, and gnfw params?
        self.z=self.z
        #self.nu=150. #same for these two lines, name these more specific things in the yaml file
        #self.M= 10**13. #Msol
        self.params_rho=[rho0,xc,bt,1] #look in example_mopc.py for these equations
        self.params_pth=[P0,al,bt,1] #look in example_mopc for these equations

        x,ksz_data,tsz_data,dy_ksz,dy_tsz=self._get_data()
        #add in sr2sqarcmin for units depending on dy we use (if we use ACT cov files)
        #obviously won't need if dy is actually the different between data and theory
        #the cov used here I think should be the actual covariance matrix?
        cov=np.diag(dy**2)
        self.data=GaussianData("SZModel",x,y,cov) #I don't think the name here matters

        #maybe define all of the stuff prior to integrating pth/rho_gnfw funcs?
        #this would be if we don't use the function set up as it is right now
        disc_fac = np.sqrt(2)
        l0 = 30000.0
        NNR = 100
        NNR2 = 3 * NNR

        AngDis = AngDist(z) #go through these functions to find cobaya equivalents

        rvir = r200(M, z) / kpc_cgs / 1e3 #replace this function at some point

        r_ext = AngDis * np.arctan(np.radians(tht / 60.0))
        r_ext2 = AngDis * np.arctan(np.radians(tht * disc_fac / 60.0))

        rvir_arcmin = 180.0 * 60.0 / np.pi * np.tan(rvir / AngDis) 
        rvir_ext = AngDis * np.arctan(np.radians(rvir_arcmin / 60.0))
        rvir_ext2 = AngDis * np.arctan(np.radians(rvir_arcmin * disc_fac / 60.0))

        rad = np.logspace(-3, 1, 2e2) 
        rad2 = np.logspace(-3, 1, 2e2) 

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


    def _get_data(self,**params_values):
        x,ksz_data,tsz_data=np.loadtxt(self.auto_file) #how do we call these in?
        self.ksz_data= #??
        self.tsz_data= #??
        self.dy_ksz= #??
        self.dy_tsz= #??
        return x,ksz_data,tsz_data,dy_ksz,dy_tsz

    def logp(self,**params_values):
        theory=self._get_theory(**params_values)
        return self.data.loglike(theory)

    def _get_theory(self,**params_values):

        #define thta_arc=x somewhere
        rho = np.zeros(len(thta_arc))
        pth = np.zeros(len(thta_arc))
        for ii in range(len(thta_arc)):
            temp = project_prof_beam(thta_arc[ii], self.M, self.z, params_rho, params_pth, nu, fbeam)
            rho[ii] = temp[0]
            pth[ii] = temp[1]

        #can "vectorize" but still need to call the project_prof function
        #vfunc=np.vectorize(project_prof_beam)
        #rho,pth=vfunc() #no theta_rho as input, see test_vectorize.py

        return (projections)

