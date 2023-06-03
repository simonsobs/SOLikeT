import numpy as np


def create_synthetic_beam(fwhm=1.4):
    """Returns beam of appropriate shape to use in mopc"""
    ...


"""from pixell.utils.beam_transform_to_profile"""


def beam_transform_to_profile(bl, theta, normalize=False):
    """Given the transform b(l) of a beam, evaluate its real space
    angular profile
    at the given radii theta."""
    bl = np.asarray(bl)
    ell = np.arange(bl.size)
    x = np.cos(theta)
    a = bl * (2 * ell + 1) / (4 * np.pi)
    profile = np.polynomial.legendre.legval(x, a)
    if normalize:
        profile /= np.sum(a)
    return profile


def read_beam(filename, normalize=True):
    """Reads the beam multipole info from file, returns real-space beam data"""
    ell, b_ell = np.genfromtxt(filename, unpack=True)
    theta_arcmin = np.linspace(0, 20, 100)  # arcmin
    theta_rad = theta_arcmin / 60.0 * (np.pi / 180)  # radians
    b = beam_transform_to_profile(b_ell, theta_rad)

    if normalize:
        integ = np.trapz(b * 2 * np.pi * theta_rad, x=theta_rad)
        b /= integ

    return b


def f_beam(tht, b):
    """Interpolates beam transform, b, from read_beam for a given theta value"""

    theta_arcmin = np.linspace(0, 20, 100)
    theta_rad = theta_arcmin / 60.0 * (np.pi / 180)

    tht_in = theta_rad

    return np.interp(tht, tht_in, b, period=np.pi)


def f_beam_fft(beam_txt, ell):
    """Reads in beam from txt file in ell, interpolates
    Specific to ACT/CMASS
    """
    beam_file = np.genfromtxt(beam_txt)
    ell_data = beam_file[:, 0]
    beam_data = beam_file[:, 1]

    return np.interp(ell, ell_data, beam_data)


def f_response(beam_response_txt, ell):
    """Reads in response from txt file in ell, interpolates
    Specific to ACT/CMASS
    """
    response_file = np.genfromtxt(beam_response_txt)
    ell_data = response_file[:, 0]
    resp_data = response_file[:, 1]

    return np.interp(ell, ell_data, resp_data)
