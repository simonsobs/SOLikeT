import numpy as np
import pdb
from ..constants import C_HMPC


oneover_chmpc = 1. / C_HMPC


def mag_bias_kernel(provider, dndz, s1, zatchi, chi_arr, chiprime_arr, zprime_arr):
    '''Calculates magnification bias kernel.

    '''
    dndzprime = np.interp(zprime_arr, dndz[:, 0], dndz[:, 1], left=0, right=0)
    norm = np.trapz(dndz[:, 1], x=dndz[:, 0])
    dndzprime = dndzprime / norm #TODO check this norm is right

    g_integrand = (chiprime_arr - chi_arr[np.newaxis, :]) / chiprime_arr \
                    * (oneover_chmpc * provider.get_param('H0') / 100) \
                    * np.sqrt(provider.get_param('omegam') * (1 + zprime_arr)**3.
                                + 1 - provider.get_param('omegam')) \
                    * dndzprime

    g = chi_arr * np.trapz(g_integrand, x=chiprime_arr, axis=0)

    W_mu = (5. * s1 - 2.) * 1.5 * provider.get_param('omegam') \
                * (provider.get_param('H0') / 100)**2 * (oneover_chmpc)**2 \
                * (1. + zatchi(chi_arr)) * g

    return W_mu


def do_limber(ell_arr, provider, dndz1, dndz2, s1, s2, pk, b1_HF, b2_HF,
              alpha_auto, alpha_cross,
              chi_grids,
              use_zeff=True, autoCMB=False,
              Nchi=50, dndz1_mag=None, dndz2_mag=None, normed=False):

    zatchi = chi_grids['zatchi']
    chiatz = chi_grids['chiatz']
    chi_arr = chi_grids['chival']
    z_arr = chi_grids['zval']
    chiprime_arr = chi_grids['chivalp']
    zprime_arr = chi_grids['zvalp']

    chistar = provider.get_comoving_radial_distance(provider.get_param('zstar'))

    # Galaxy kernels, assumed to be b(z) * dN/dz
    W_g1 = np.interp(zatchi(chi_arr), dndz1[:, 0], dndz1[:, 1] \
                * provider.get_Hubble(dndz1[:, 0], units='1/Mpc'), left=0, right=0)
    if not normed:
        W_g1 /= np.trapz(W_g1, x=chi_arr)

    W_g2 = np.interp(zatchi(chi_arr), dndz2[:, 0], dndz2[:, 1] \
                * provider.get_Hubble(dndz2[:, 0], units='1/Mpc'), left=0, right=0)
    if not normed:
        W_g2 /= np.trapz(W_g2, x=chi_arr)

    W_kappa = (oneover_chmpc)**2. * 1.5 * provider.get_param('omegam') \
                * (provider.get_param('H0') / 100)**2. * (1. + zatchi(chi_arr)) \
                * chi_arr * (chistar - chi_arr) / chistar

    # Get effective redshift
    # if use_zeff:
    #     kern = W_g1 * W_g2 / chi_arr**2
    #     zeff = np.trapz(kern * z_arr,x=chi_arr) / np.trapz(kern, x=chi_arr)
    # else:
    #     zeff = -1.0

    # set up magnification bias kernels
    W_mu1 = mag_bias_kernel(provider, dndz1, s1, zatchi, chi_arr, chiprime_arr, zprime_arr)

    c_ell_g1g1 = np.zeros([ell_arr.shape[0], 1, chi_arr.shape[0]])
    c_ell_g1kappa = np.zeros([ell_arr.shape[0], 1, chi_arr.shape[0]])
    c_ell_kappakappa = np.zeros([ell_arr.shape[0], 1, chi_arr.shape[0]])

    c_ell_g1mu1 = np.zeros([ell_arr.shape[0], 1, chi_arr.shape[0]])
    c_ell_mu1mu1 = np.zeros([ell_arr.shape[0], 1, chi_arr.shape[0]])
    c_ell_mu1kappa = np.zeros([ell_arr.shape[0], 1, chi_arr.shape[0]])

    for i_chi, chi in enumerate(chi_arr):

        k_arr = (ell_arr + 0.5) / chi

        p_mm_hf = pk(zatchi(chi), k_arr)
        p_mm = p_mm_hf
        p_gg = b1_HF * b1_HF * p_mm_hf # lets just stay at constant linear bias for now
        p_gm = b1_HF * p_mm_hf

        W_g1g1 = W_g1[i_chi] * W_g1[i_chi] / (chi**2) * p_gg
        c_ell_g1g1[:, :, i_chi] = W_g1g1.T

        W_g1kappa = W_g1[i_chi] * W_kappa[i_chi] / (chi**2) * p_gm
        c_ell_g1kappa[:, :, i_chi] = W_g1kappa.T

        # W_kappakappa = W_kappa[i_chi] * W_kappa[i_chi] / (chi**2) * p_mm
        # c_ell_kappakappa[:,:,i_chi] = W_kappakappa.T

        W_g1mu1 = W_g1[i_chi] * W_mu1[i_chi] / (chi**2) * p_gm
        c_ell_g1mu1[:, :, i_chi] = W_g1mu1.T

        W_mu1mu1 = W_mu1[i_chi] * W_mu1[i_chi] / (chi**2) * p_mm
        c_ell_mu1mu1[:, :, i_chi] = W_mu1mu1.T

        W_mu1kappa = W_kappa[i_chi] * W_mu1[i_chi] / (chi**2) * p_mm
        c_ell_mu1kappa[:, :, i_chi] = W_mu1kappa.T


    c_ell_g1g1 = np.trapz(c_ell_g1g1, x=chi_arr, axis=-1)
    c_ell_g1kappa = np.trapz(c_ell_g1kappa, x=chi_arr, axis=-1)
    c_ell_kappakappa = np.trapz(c_ell_kappakappa, x=chi_arr, axis=-1)

    c_ell_g1mu1 = np.trapz(c_ell_g1mu1, x=chi_arr, axis=-1)
    c_ell_mu1mu1 = np.trapz(c_ell_mu1mu1, x=chi_arr, axis=-1)
    c_ell_mu1kappa = np.trapz(c_ell_mu1kappa, x=chi_arr, axis=-1)

    clobs_gg = c_ell_g1g1 + 2. * c_ell_g1mu1 + c_ell_mu1mu1
    clobs_kappag = c_ell_g1kappa + c_ell_mu1kappa
    # clobs_kappakappa = c_ell_kappakappa

    return clobs_gg.flatten(), clobs_kappag.flatten()#, clobs_kappakappa.flatten()
