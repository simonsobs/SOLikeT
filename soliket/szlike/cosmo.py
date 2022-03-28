import numpy as np
from scipy.integrate import quad
from .params import cosmo_params

rhocrit =  1.87847e-29 * cosmo_params['hh']**2


def hub_func(z):
    '''Hubble function
    '''
    Om = cosmo_params['Omega_m']
    Ol = cosmo_params['Omega_L']
    O_tot = Om + Ol
    ans = np.sqrt(Om*(1.0 + z)**3 + Ol + (1 - O_tot)*(1 + z)**2)
    return ans

def rho_cz(z):
    '''critical density in cgs
    '''
    Ez2 = cosmo_params['Omega_m']*(1+z)**3. + (1-cosmo_params['Omega_m'])
    return rhocrit * Ez2


def ComInt(z):
    '''comoving distance integrand
    '''
    ans = 1.0/hub_func(z)
    return ans


def ComDist(z):
    '''comoving distance
    '''
    Om = cosmo_params['Omega_m']
    Ol = cosmo_params['Omega_L']
    O_tot = Om + Ol
    Dh = cosmo_params['C_OVER_HUBBLE']/cosmo_params['hh']
    ans = Dh*quad(ComInt,0,z)[0]
    if (O_tot < 1.0): ans = Dh / np.sqrt(1.0-O_tot) *  np.sin(np.sqrt(1.0-O_tot) * quad(ComInt,0,z)[0])
    if (O_tot > 1.0): ans = Dh / np.sqrt(O_tot-1.0) *  np.sinh(np.sqrt(O_tot-1.0) * quad(ComInt,0,z)[0])
    return ans

def AngDist(z):
    '''angular distance
    '''
    ans = ComDist(z)/(1.0+z)
    return ans