import numpy as np
import pdb

oneover_chmpc = 1./2997.92458

def pk_dummy(dummy_halofit, kval,zz,b1_HF,b2_HF,one_loop=True,shear=True):
        # TODO remove this!
        """Returns power spectra at k=kval and z=zz.
        pars is a dict telling you which method to use (halozeldovich, zeft, halofit)
        and the bias/1-halo parameters for 2 tracers.  
        ex: pars = {'method': 'halofit', 'tracer1': {'b': lambda z: 1.5},
        'tracer2': {'b': lambda z: 2.0}}.
        Note that biases must be functions of redshift.
        This function then returns all 4 spectra needed for CMB lensing:
        tracer1 x tracer2 , tracer1 x matter, tracer2 x matter, and matter x matter.
        If you want auto-spectra, just use the same bias coefficients for tracer1 and tracer2."""
        # Get the interpolating coefficients then just weigh each P(k,z).
        # t0 = time.time()
        
        p_mm = np.zeros((1,len(kval)))
        p_gm_HF = np.zeros((1,len(kval)))
        p_gg_HF = np.zeros((1,len(kval)))

        p_mm[0,:] = dummy_halofit(zz, kval)
        p_gm_HF[0,:] = dummy_halofit(zz, kval)
        p_gg_HF[0,:] = dummy_halofit(zz, kval)

        return(p_gg_HF, p_gm_HF, p_mm)

def mag_bias_kernel(cosmo, dndz, s1, zatchi, chi_arr, chiprime_arr, zprime_arr):

        dndzprime = np.interp(zprime_arr, dndz[:,0], dndz[:,1], left=0, right=0)
        norm = np.trapz(dndz[:,1], x=dndz[:,0])
        dndzprime  = dndzprime/norm #TODO check this norm is right

        g_integrand = (chiprime_arr - chi_arr[np.newaxis,:]) / chiprime_arr * np.sqrt(cosmo.get_param('omegam') * (1 + zprime_arr)**3. + 1 - cosmo.get_param('omegam') ) * dndzprime

        g = chi_arr * np.trapz(g_integrand, x=chiprime_arr, axis=0)

        W_mu = (5. * s1 - 2.) * 1.5 * cosmo.get_param('omegam') * (oneover_chmpc)**2 * (1. + zatchi(chi_arr)) * g

        return W_mu

def do_limber(ell_arr, cosmo, dndz1, dndz2, s1, s2, pk, b1_HF, b2_HF, alpha_auto, alpha_cross, use_zeff=True, autoCMB=False, Nchi=50, dndz1_mag=None, dndz2_mag=None, normed=False, setup_chi_flag=False, setup_chi_out=None):
    
    zatchi, chiatz, chi_arr, z_arr, chiprime_arr, zprime_arr = setup_chi_out
    chistar = cosmo.get_comoving_radial_distance(cosmo.get_param('zstar'))

    # Galaxy kernels, assumed to be b(z) * dN/dz
    W_g1    = np.interp(zatchi(chi_arr), dndz1[:,0], dndz1[:,1]*cosmo.get_Hubble(dndz1[:,0], units='1/Mpc'), left=0, right=0)
    if not normed:
        W_g1   /= np.trapz(W_g1,x=chi_arr)

    W_g2    = np.interp(zatchi(chi_arr), dndz2[:,0], dndz2[:,1]*cosmo.get_Hubble(dndz2[:,0], units='1/Mpc'), left=0, right=0)
    if not normed:
        W_g2   /= np.trapz(W_g2,x=chi_arr)

    W_kappa = (oneover_chmpc)**2. * 1.5 * cosmo.get_param('omegam') * (cosmo.get_param('H0') / 100)**2. \
                * (1. + zatchi(chi_arr)) * chi_arr * (chistar - chi_arr) / chistar

    # Get effective redshift     
    if use_zeff:
        kern = W_g1*W_g2/chi_arr**2
        zeff = np.trapz(kern*z_arr,x=chi_arr)/np.trapz(kern,x=chi_arr)
    else:
        zeff = -1.0

    # set up magnification bias kernels
    W_mu1 = mag_bias_kernel(cosmo, dndz1, s1, zatchi, chi_arr, chiprime_arr, zprime_arr)

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
        c_ell_g1g1[:,:,i_chi] = W_g1g1.T

        W_g1kappa = W_g1[i_chi] * W_kappa[i_chi] / (chi**2) * p_gm
        c_ell_g1kappa[:,:,i_chi] = W_g1kappa.T

        # W_kappakappa = W_kappa[i_chi] * W_kappa[i_chi] / (chi**2) * p_mm
        # c_ell_kappakappa[:,:,i_chi] = W_kappakappa.T

        W_g1mu1 = W_g1[i_chi]*W_mu1[i_chi] / chi**2 * p_gm
        c_ell_g1mu1[:,:,i_chi] = W_g1mu1.T

        W_mu1mu1 = W_mu1[i_chi]*W_mu1[i_chi] / chi**2 * p_mm
        c_ell_mu1mu1[:,:,i_chi] = W_mu1mu1.T

        W_mu1kappa = W_kappa[i_chi]*W_mu1[i_chi] / chi**2 * p_mm
        c_ell_mu1kappa[:,:,i_chi] = W_mu1kappa.T


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
