import numpy as np
import scipy
import scipy.integrate
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline as _spline
from scipy import special
import scipy.constants as con
from astropy.cosmology import Planck18
import astropy.units as u
from numpy import trapz

cosmo = Planck18

class u_p_nfw_hmf_bias:
    def __init__(self, k_array, Pk_array, mh, redshift, delta_h):
        self.k_array = k_array
        self.Pk_array = Pk_array
        self.mh = mh
        self.redshift = redshift
        self.delta_h = delta_h
        self.compute_nfw()
        self.compute_hmf()
        self.compute_b_CIB()


    def mean_density(self):
        mean_density0 = (
            (cosmo.Om0 * cosmo.critical_density0).to(u.Msun / u.Mpc ** 3).value
        )
        mean_density0 /= cosmo.h ** 2
        return mean_density0

    # Lagrangian radius
    def mass_to_radius(self):
        rho_mean = self.mean_density()
        r3 = 3 * self.mh / (4 * np.pi * rho_mean)
        return r3 ** (1.0 / 3.0)

    # virial radius
    def r_delta(self, red):
        rho_crit = self.mean_density() * (1 + red) ** 3
        r3 = 3 * self.mh / (4 * np.pi * self.delta_h * rho_crit)
        return (1 + red) * r3 ** (1.0 / 3.0)

    # Fourier transform of top hat window function
    def W(self, rk):
        return np.where(rk > 1.4e-6, (3 * (np.sin(rk) - rk * np.cos(rk)) / rk ** 3), 1)

    def sigma(self, rad, red, zeta):
        k = self.k_array
        P_linear = self.Pk_array[zeta]
        rk = np.outer(rad, k)
        rest = P_linear * k ** 3
        lnk = np.log(k)
        uW = self.W(rk)
        integ = rest * uW ** 2
        sigm = (0.5 / np.pi ** 2) * scipy.integrate.simps(integ, x=lnk, axis=-1)
        return np.sqrt(sigm)

    # sigma depends on z from the linear power spectrum
    def nu_delta(self, red, zeta):
        rad = self.mass_to_radius()
        delta_c = 1.686  # critical overdensity of the universe.
        sig = self.sigma(rad, red, zeta)
        return delta_c / sig

    def nu_to_c200c(self, red, zeta):
        nu = self.nu_delta(red, zeta)
        diff = np.abs(nu - 1)
        ind_M_star = np.where(diff == min(diff))[0][0]
        M_star = self.mh[ind_M_star]
        a = 9
        b = -0.13
        conc = a / (1 + red) * (self.mh / M_star) ** b
        return conc

    def r_star(self, red, zeta):
        c_200c = self.nu_to_c200c(red, zeta)
        r200 = self.r_delta(red)
        return r200 / c_200c

    def ampl_nfw(self, c):
        return 1.0 / (np.log(1 + c) - c / (1 + c))

    def sine_cosine_int(self, x):
        si, ci = scipy.special.sici(x)
        return si, ci

    def nfwfourier_u(self, red, zeta):
        k = self.k_array
        rs = self.r_star(red, zeta)
        c = self.nu_to_c200c(red, zeta)
        a = self.ampl_nfw(c)
        mu = np.outer(rs, k)
        Si1, Ci1 = self.sine_cosine_int(mu + mu * c[:, np.newaxis])
        Si2, Ci2 = self.sine_cosine_int(mu)
        unfw = a[:, np.newaxis] * (
            np.cos(mu) * (Ci1 - Ci2)
            + np.sin(mu) * (Si1 - Si2)
            - np.sin(mu * c[:, np.newaxis]) / (mu + mu * c[:, np.newaxis])
        )
        return unfw

    def compute_nfw(self):
        u_k = np.zeros([len(self.redshift), len(self.mh), len(self.k_array)])
        for zeta in range(len(self.redshift)):
            red = self.redshift[zeta]
            u_k[zeta] = self.nfwfourier_u(red, zeta)

        self.u_k = u_k

        return u_k                    

    def dw_dlnkr(self, rk):
        return np.where(
            rk > 1e-3,
            (9 * rk * np.cos(rk) + 3 * np.sin(rk) * (rk ** 2 - 3)) / rk ** 3,
            0,
        )

    def dlns2_dlnr(self, rad, red, zeta):
        k = self.k_array
        P_linear = self.Pk_array[zeta]
        rk = np.outer(rad, k)
        rest = P_linear * k ** 3
        w = self.W(rk)
        dw = self.dw_dlnkr(rk)
        inte = w * dw * rest
        lnk = np.log(k)
        s = self.sigma(rad, red, zeta)
        return scipy.integrate.simps(inte, x=lnk, axis=-1, even="avg") / (
            np.pi ** 2 * s ** 2
        )

    def dlnr_dlnm(self):
        return 1.0 / 3.0

    def dlns2_dlnm(self, rad, red, zeta):
        return self.dlns2_dlnr(rad, red, zeta) * self.dlnr_dlnm()

    def dlns_dlnm(self, rad, red, zeta):
        return 0.5 * self.dlns2_dlnm(rad, red, zeta)

    def coefficient(self, dhalo):
        a = {  # -- A
            "A_200": 1.858659e-01,
            "A_300": 1.995973e-01,
            "A_400": 2.115659e-01,
            "A_600": 2.184113e-01,
            "A_800": 2.480968e-01,
            "A_1200": 2.546053e-01,
            "A_1600": 2.600000e-01,
            "A_2400": 2.600000e-01,
            "A_3200": 2.600000e-01,
            # -- a
            "a_200": 1.466904,
            "a_300": 1.521782,
            "a_400": 1.559186,
            "a_600": 1.614585,
            "a_800": 1.869936,
            "a_1200": 2.128056,
            "a_1600": 2.301275,
            "a_2400": 2.529241,
            "a_3200": 2.661983,
            # --- b
            "b_200": 2.571104,
            "b_300": 2.254217,
            "b_400": 2.048674,
            "b_600": 1.869559,
            "b_800": 1.588649,
            "b_1200": 1.507134,
            "b_1600": 1.464374,
            "b_2400": 1.436827,
            "b_3200": 1.405210,
            # --- c
            "c_200": 1.193958,
            "c_300": 1.270316,
            "c_400": 1.335191,
            "c_600": 1.446266,
            "c_800": 1.581345,
            "c_1200": 1.795050,
            "c_1600": 1.965613,
            "c_2400": 2.237466,
            "c_3200": 2.439729,
        }

        delta_virs = np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200])

        A_array = np.array([a["A_%s" % d] for d in delta_virs])
        a_array = np.array([a["a_%s" % d] for d in delta_virs])
        b_array = np.array([a["b_%s" % d] for d in delta_virs])
        c_array = np.array([a["c_%s" % d] for d in delta_virs])

        A_intfunc = _spline(delta_virs, A_array)
        a_intfunc = _spline(delta_virs, a_array)
        b_intfunc = _spline(delta_virs, b_array)
        c_intfunc = _spline(delta_virs, c_array)

        A_0 = A_intfunc(dhalo)
        a_0 = a_intfunc(dhalo)
        b_0 = b_intfunc(dhalo)
        c_0 = c_intfunc(dhalo)
        return A_0, a_0, b_0, c_0

    def fsigma(self, rad, red, zeta):
        z = red
        dhalo = self.delta_h / cosmo.Om(z)
        lgdelta = np.log10(dhalo)
        A_0 = self.coefficient(dhalo)[0]
        a_0 = self.coefficient(dhalo)[1]
        b_0 = self.coefficient(dhalo)[2]
        c_0 = self.coefficient(dhalo)[3]
        A_exp = -0.14
        a_exp = -0.06
        s = self.sigma(rad, red, zeta)
        A = A_0 * (1 + z) ** A_exp
        a = a_0 * (1 + z) ** a_exp
        alpha = 10 ** (-((0.75 / np.log10(dhalo / 75.0)) ** 1.2))
        b = b_0 * (1 + z) ** (-alpha)
        return A * ((s / b) ** (-a) + 1) * np.exp(-c_0 / s ** 2)

    def dn_dm(self, red, zeta):
        rad = self.mass_to_radius()
        return self.fsigma(rad, red, zeta) * self.mean_density() * np.abs(self.dlns_dlnm(rad, red, zeta)) / self.mh**2

    def dn_dlnm(self, red, zeta):
        return self.mh * self.dn_dm(red, zeta)

    def dn_dlogm(self, red, zeta):
        return self.mh * self.dn_dm(red, zeta) * np.log(10)

    def compute_hmf(self):
        dndM = np.zeros([len(self.redshift), len(self.mh)])
        for zeta in range(len(self.redshift)):
            red = self.redshift[zeta]
            dndM[zeta] = self.dn_dm(red, zeta)

        self.dndM = dndM

        return dndM

    def b_CIB(self, red, zeta):
        rad = self.mass_to_radius()
        A = 1.04
        aa = 0.132
        B = 0.183
        b = 1.5
        C = 0.262
        c = 2.4
        s = self.sigma(rad, red, zeta)
        nuu = 1.686 / s
        dc = 1.686  # neglecting the redshift evolution
        return (
            1
            - (A * nuu ** aa / (nuu ** aa + dc ** aa))
            + B * nuu ** b
            + C * nuu ** c
        )

    def compute_b_CIB(self):
        bias = np.zeros([len(self.redshift), len(self.mh)])
        for zeta in range(len(self.redshift)):
            red = self.redshift[zeta]
            bias[zeta] = self.b_CIB(red, zeta)

        self.bias_cib = bias
        
        return bias