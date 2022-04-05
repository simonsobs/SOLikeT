import numpy as np
from scipy.integrate import quad

rhocrit =  1.87847e-29 * provider.get_param('hh')**2


def hub_func(z):
    '''Hubble function
    '''
    Om = provider.get_param('Omega_m')
    Ol = provider.get_param('Omega_L')
    O_tot = Om + Ol
    ans = np.sqrt(Om*(1.0 + z)**3 + Ol + (1 - O_tot)*(1 + z)**2)
    return ans

def rho_cz(z):
    '''critical density in cgs
    '''
    Ez2 = provider.get_param('Omega_m')*(1+z)**3. + (1-provider.get_param('Omega_m'))
    return rhocrit * Ez2


def ComInt(z):
    '''comoving distance integrand
    '''
    ans = 1.0/hub_func(z)
    return ans


def ComDist(z):
    '''comoving distance
    '''
    Om = provider.get_param('Omega_m')
    Ol = provider.get_param('Omega_L')
    O_tot = Om + Ol
    Dh = provider.get_param('C_OVER_HUBBLE')/provider.get_param('hh')
    ans = Dh*quad(ComInt,0,z)[0]
    if (O_tot < 1.0): ans = Dh / np.sqrt(1.0-O_tot) *  np.sin(np.sqrt(1.0-O_tot) * quad(ComInt,0,z)[0])
    if (O_tot > 1.0): ans = Dh / np.sqrt(O_tot-1.0) *  np.sinh(np.sqrt(O_tot-1.0) * quad(ComInt,0,z)[0])
    return ans

def AngDist(z):
    '''angular distance
    '''
    ans = ComDist(z)/(1.0+z)
    return ans