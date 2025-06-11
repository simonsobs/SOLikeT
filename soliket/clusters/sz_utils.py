"""
.. module:: szutils

Contains functions (many inherited from the
`nemo <https://nemo-sz.readthedocs.io/en/latest/>`_) code) which are used internally by
the cluster likelihood to convert between observed tSZ signal and cluster mass.

"""

import numpy as np

try:
    from numpy import trapezoid
except ImportError:
    from numpy import trapz as trapezoid
from scipy import interpolate

# from nemo import signals
from soliket.constants import (
    C_M_S,
    G_CGS,
    MPC2CM,
    MSUN_CGS,
    T_CMB,
    electron_mass_kg,
    elementary_charge,
    h_Planck,
    k_Boltzmann,
)

# from astropy.cosmology import FlatLambdaCDM
# from .clusters import C_KM_S as C_in_kms

rho_crit0H100 = (3.0 / (8.0 * np.pi) * (100.0 * 1.0e5) ** 2.0) / G_CGS * MPC2CM / MSUN_CGS


def gaussian(
    x: np.ndarray, mean: float, sigma: float, no_norm: bool = False
) -> np.ndarray:
    """Return a Gaussian or unnormalized Gaussian evaluated at x.

    Args:
        x: Input array.
        mean: Mean of the Gaussian.
        sigma: Standard deviation.
        no_norm: If True, do not normalize the output.

    Returns:
        Gaussian evaluated at x.
    """
    if no_norm:
        return np.exp(-1.0 * (x - mean) ** 2 / (2.0 * sigma**2.0))
    else:
        return (
            1.0
            / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-1.0 * (x - mean) ** 2 / (2.0 * sigma**2.0))
        )


class SZUtils:
    """
    Helper functions for tSZ signal and cluster mass conversions for cluster likelihoods.

    Many methods are adapted from the nemo-sz codebase.
    The Survey object should provide attributes like Q (filter spline)
    and qmin (detection threshold).
    """

    def __init__(self, survey):
        """Initialize with a survey object providing Q and qmin attributes."""
        self.LgY = np.arange(-6, -2.5, 0.01)
        self.survey = survey
        # self.rho_crit0H100 = (
        #   (3. / (8. * np.pi) * (100. * 1.e5)**2.) / G_in_cgs * Mpc_in_cm / MSun_in_g
        # )

    def p_y_given_mass(
        self,
        log_y: np.ndarray,
        mass: np.ndarray,
        z: np.ndarray,
        params: dict,
        ez_func,
        da_func,
    ) -> np.ndarray:
        """
        Probability density P(Y|M,z) for tSZ signal Y given mass and redshift.
        Args:
            log_y: Log10(Y) grid.
            mass: Mass array.
            z: Redshift array.
            params: Model parameters (must include 'H0', 'massbias', 'scat', 'B0').
            ez_func: Function Ez(z).
            da_func: Function Da(z).
        Returns:
            Probability density array.
        """
        H0 = params["H0"]
        mass_grid = np.outer(mass, np.ones(len(log_y[0, :])))
        # y0_from_logm500 returns y_tilde, theta0, q_filt
        y_tilde, theta0, q_filt = y0_from_logm500(
            np.log10(params["massbias"] * mass_grid / (H0 / 100.0)),
            z,
            self.survey.Q,
            sigma_int=params["scat"],
            B0=params["B0"],
            H0=params["H0"],
            Ez_fn=ez_func,
            Da_fn=da_func,
        )
        Y = 10**log_y
        exponent = -1.0 * (np.log(Y / y_tilde)) ** 2
        return (
            1.0
            / (params["scat"] * np.sqrt(2 * np.pi))
            * np.exp(exponent / (2.0 * params["scat"] ** 2))
        )

    def p_y_given_mass_vec(
        self,
        log_y: np.ndarray,
        mass: np.ndarray,
        z: np.ndarray,
        params: dict,
        ez_func,
        da_func,
    ) -> np.ndarray:
        """
        Vectorized version of p_y_given_mass for 3D log_y arrays.
        """
        H0 = params["H0"]
        y_tilde, theta0, q_filt = y0_from_logm500(
            np.log10(params["massbias"] * mass / (H0 / 100.0)),
            z,
            self.survey.Q,
            sigma_int=params["scat"],
            B0=params["B0"],
            H0=params["H0"],
            Ez_fn=ez_func,
            Da_fn=da_func,
        )
        Y = 10**log_y
        y_tilde = np.repeat(y_tilde[:, :, np.newaxis], log_y.shape[2], axis=2)
        exponent = -1.0 * (np.log(Y / y_tilde)) ** 2
        return (
            1.0
            / (params["scat"] * np.sqrt(2 * np.pi))
            * np.exp(exponent / (2.0 * params["scat"] ** 2))
        )

    def detection_mask(self, Y: np.ndarray, Y_noise: np.ndarray) -> np.ndarray:
        """
        Return a mask array where Y exceeds the survey detection threshold.
        """
        qmin = self.survey.qmin
        mask = np.zeros_like(Y)
        mask[Y - qmin * Y_noise > 0] = 1.0
        return mask

    def p_y_above_sn(
        self,
        log_y: np.ndarray,
        mass: np.ndarray,
        z: np.ndarray,
        Y_noise: np.ndarray,
        params: dict,
        ez_func,
        da_func,
    ) -> np.ndarray:
        """
        Probability of Y above S/N threshold for given mass and redshift.
        """
        Y = 10**log_y
        mask = self.detection_mask(Y, Y_noise)
        mask_reshaped = np.reshape(
            np.outer(np.ones([mass.shape[0], mass.shape[1]]), mask),
            (mass.shape[0], mass.shape[1], len(mask)),
        )
        log_y_grid = np.outer(np.ones([mass.shape[0], mass.shape[1]]), log_y)
        log_y_grid_reshaped = np.reshape(
            log_y_grid, (mass.shape[0], mass.shape[1], len(log_y))
        )
        p_y = np.nan_to_num(
            self.p_y_given_mass_vec(
                log_y_grid_reshaped, mass, z, params, ez_func, da_func
            )
        )
        # Use numpy.trapezoid for integration (requires numpy >=1.17)
        result = trapezoid(p_y * mask_reshaped, x=log_y, axis=2) * np.log(10)
        return result

    def pfunc_y(
        self,
        Y_noise: np.ndarray,
        mass: np.ndarray,
        z_arr: np.ndarray,
        params: dict,
        ez_func,
        da_func,
    ) -> np.ndarray:
        """
        Return probability function for Y above S/N for all masses and redshifts.
        """
        log_y = self.LgY
        mass_grid = np.outer(mass, np.ones([len(z_arr)]))
        return self.p_y_above_sn(
            log_y, mass_grid, z_arr, Y_noise, params, ez_func, da_func
        )

    def p_y_per_observed(
        self,
        log_y: np.ndarray,
        mass: np.ndarray,
        z: np.ndarray,
        Y_c: float,
        Y_err: float,
        params: dict,
    ) -> np.ndarray:
        """
        Probability for observed Y given mass and redshift, marginalized over
        measurement error.
        """
        p_y_sig = np.outer(np.ones(len(mass)), self.y_prob(Y_c, log_y, Y_err))
        log_y_grid = np.outer(np.ones([mass.shape[0], mass.shape[1]]), log_y)
        log_y_grid_reshaped = np.reshape(
            log_y_grid, (mass.shape[0], mass.shape[1], len(log_y))
        )
        p_y = np.nan_to_num(self.p_y_given_mass(log_y_grid_reshaped, mass, z, params))
        # Use numpy.trapezoid for integration (requires numpy >=1.17)
        result = trapezoid(p_y * p_y_sig, log_y, np.diff(log_y), axis=1) * np.log(10)
        return result

    def y_prob(self, Y_c: float, log_y: np.ndarray, Y_noise: float) -> np.ndarray:
        """
        Return Gaussian probability for observed Y_c given log_y and noise.
        """
        y_val = 10**log_y
        return gaussian(y_val, Y_c, Y_noise)

    def pfunc_per(
        self,
        mass: np.ndarray,
        z: np.ndarray,
        Y_c: float,
        Y_err: float,
        params: dict,
        ez_func,
        da_func,
    ) -> np.ndarray:
        """
        Probability marginalized over measurement error for all masses and redshifts.
        """
        log_y = self.LgY
        log_y_grid = np.outer(np.ones(len(mass)), log_y)
        p_y_sig = self.y_prob(Y_c, log_y, Y_err)
        p_y = np.nan_to_num(
            self.p_y_given_mass(log_y_grid, mass, z, params, ez_func, da_func)
        )
        # Use numpy.trapezoid for integration (requires numpy >=1.17)
        result = trapezoid(p_y * p_y_sig, log_y, np.diff(log_y), axis=1)
        return result

    def pfunc_per_parallel(
        self,
        mass: np.ndarray,
        z: np.ndarray,
        Y_c: float,
        Y_err: float,
        params: dict,
        ez_func,
        da_func,
    ) -> np.ndarray:
        """
        Parallelized version of pfunc_per for all masses and redshifts.
        """
        p_y_sig = self.y_prob(Y_c, self.LgY, Y_err)
        p_y = np.nan_to_num(
            self.p_y_given_mass(self.LgY, mass, z, params, ez_func, da_func)
        )
        # Use numpy.trapezoid for integration (requires numpy >=1.17)
        result = trapezoid(p_y * p_y_sig, x=self.LgY, axis=2)
        return result

    def pfunc_per_zarr(
        self,
        mass: np.ndarray,
        z_c: np.ndarray,
        Y_c: float,
        Y_err: float,
        int_hmf: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        Probability marginalized over measurement error for a grid of masses
        and redshifts.
        """
        log_y = self.LgY
        # old was z_arr
        # P_func = np.outer(mass, np.zeros([len(z_arr)]))
        # M_arr = np.outer(mass, np.ones([len(z_arr)]))
        # M200 = np.outer(mass, np.zeros([len(z_arr)]))
        # zarr = np.outer(np.ones([len(mass)]), z_arr)
        P_func = self.p_y_per_observed(log_y, mass, z_c, Y_c, Y_err, params)
        return P_func


###
"""Routines from nemo (author: Matt Hilton ) to limit dependencies"""


# ----------------------------------------------------------------------------------------
def calcR500Mpc(z, M500, Ez_fn, H0):
    """Given z, M500 (in MSun), returns R500 in Mpc, with respect to critical density."""

    if type(M500) == str:
        raise Exception(
            "M500 is a string - check M500MSun in your .yml config file:\
             use, e.g., 1.0e+14 (not 1e14 or 1e+14)"
        )

    Ez = Ez_fn(z)

    criticalDensity = rho_crit0H100 * (H0 / 100.0) ** 2 * Ez**2
    R500Mpc = np.power((3 * M500) / (4 * np.pi * 500 * criticalDensity), 1.0 / 3.0)

    return R500Mpc


# ----------------------------------------------------------------------------------------
def calcTheta500Arcmin(z, M500, Ez_fn, Da_fn, H0):
    """Given z, M500 (in MSun), returns angular size equivalent to R500, with respect to
    critical density.

    """

    R500Mpc = calcR500Mpc(z, M500, Ez_fn, H0)
    DAz = Da_fn(z)

    theta500Arcmin = np.degrees(np.arctan(R500Mpc / DAz)) * 60.0

    return theta500Arcmin


# ----------------------------------------------------------------------------------------
def calcQ(theta500Arcmin, tck):
    """Returns Q, given theta500Arcmin, and a set of spline fit knots for (theta, Q)."""

    # Q=np.poly1d(coeffs)(theta500Arcmin)
    Q = interpolate.splev(theta500Arcmin, tck)

    return Q


# ----------------------------------------------------------------------------------------
def calcFRel(z, M500, Ez_fn: interpolate.interp1d, obsFreqGHz=148.0):
    """Calculates relativistic correction to SZ effect at specified frequency, given z,
    M500 in MSun.

    This assumes the Arnaud et al. (2005) M-T relation, and applies formulae of
    Itoh et al. (1998)

    As for H13, we return fRel = 1 + delta_SZE (see also Marriage et al. 2011)
    """

    # Using Arnaud et al. (2005) M-T to get temperature
    A = 3.84e14
    B = 1.71
    # TkeV=5.*np.power(((cosmoModel.efunc(z)*M500)/A), 1/B)   # HMF/Astropy
    Ez = Ez_fn(z)
    TkeV = 5.0 * np.power(((Ez * M500) / A), 1 / B)  # Colossus
    TKelvin = TkeV * ((1000 * elementary_charge) / k_Boltzmann)

    # Itoh et al. (1998) eqns. 2.25 - 2.30
    thetae = (k_Boltzmann * TKelvin) / (electron_mass_kg * C_M_S**2)
    X = (h_Planck * obsFreqGHz * 1e9) / (k_Boltzmann * T_CMB)
    Xtw = X * (np.cosh(X / 2.0) / np.sinh(X / 2.0))
    Stw = X / np.sinh(X / 2.0)

    Y0 = -4 + Xtw

    Y1 = (
        -10.0
        + (47 / 2.0) * Xtw
        - (42 / 5.0) * Xtw**2
        + (7 / 10.0) * Xtw**3
        + np.power(Stw, 2) * (-(21 / 5.0) + (7 / 5.0) * Xtw)
    )

    Y2 = (
        -(15 / 2.0)
        + (1023 / 8.0) * Xtw
        - (868 / 5.0) * Xtw**2
        + (329 / 5.0) * Xtw**3
        - (44 / 5.0) * Xtw**4
        + (11 / 30.0) * Xtw**5
        + np.power(Stw, 2)
        * (
            -(434 / 5.0)
            + (658 / 5.0) * Xtw
            - (242 / 5.0) * Xtw**2
            + (143 / 30.0) * Xtw**3
        )
        + np.power(Stw, 4) * (-(44 / 5.0) + (187 / 60.0) * Xtw)
    )

    Y3 = (
        (15 / 2.0)
        + (2505 / 8.0) * Xtw
        - (7098 / 5.0) * Xtw**2
        + (14253 / 10.0) * Xtw**3
        - (18594 / 35.0) * Xtw**4
        + (12059 / 140.0) * Xtw**5
        - (128 / 21.0) * Xtw**6
        + (16 / 105.0) * Xtw**7
        + np.power(Stw, 2)
        * (
            -(7098 / 10.0)
            + (14253 / 5.0) * Xtw
            - (102267 / 35.0) * Xtw**2
            + (156767 / 140.0) * Xtw**3
            - (1216 / 7.0) * Xtw**4
            + (64 / 7.0) * Xtw**5
        )
        + np.power(Stw, 4)
        * (
            -(18594 / 35.0)
            + (205003 / 280.0) * Xtw
            - (1920 / 7.0) * Xtw**2
            + (1024 / 35.0) * Xtw**3
        )
        + np.power(Stw, 6) * (-(544 / 21.0) + (992 / 105.0) * Xtw)
    )

    Y4 = (
        -(135 / 32.0)
        + (30375 / 128.0) * Xtw
        - (62391 / 10.0) * Xtw**2
        + (614727 / 40.0) * Xtw**3
        - (124389 / 10.0) * Xtw**4
        + (355703 / 80.0) * Xtw**5
        - (16568 / 21.0) * Xtw**6
        + (7516 / 105.0) * Xtw**7
        - (22 / 7.0) * Xtw**8
        + (11 / 210.0) * Xtw**9
        + np.power(Stw, 2)
        * (
            -(62391 / 20.0)
            + (614727 / 20.0) * Xtw
            - (1368279 / 20.0) * Xtw**2
            + (4624139 / 80.0) * Xtw**3
            - (157396 / 7.0) * Xtw**4
            + (30064 / 7.0) * Xtw**5
            - (2717 / 7.0) * Xtw**6
            + (2761 / 210.0) * Xtw**7
        )
        + np.power(Stw, 4)
        * (
            -(124389 / 10.0)
            + (6046951 / 160.0) * Xtw
            - (248520 / 7.0) * Xtw**2
            + (481024 / 35.0) * Xtw**3
            - (15972 / 7.0) * Xtw**4
            + (18689 / 140.0) * Xtw**5
        )
        + np.power(Stw, 6)
        * (
            -(70414 / 21.0)
            + (465992 / 105.0) * Xtw
            - (11792 / 7.0) * Xtw**2
            + (19778 / 105.0) * Xtw**3
        )
        + np.power(Stw, 8) * (-(682 / 7.0) + (7601 / 210.0) * Xtw)
    )

    deltaSZE = (
        ((X**3) / (np.exp(X) - 1))
        * ((thetae * X * np.exp(X)) / (np.exp(X) - 1))
        * (Y0 + Y1 * thetae + Y2 * thetae**2 + Y3 * thetae**3 + Y4 * thetae**4)
    )

    fRel = 1 + deltaSZE

    return fRel


# ----------------------------------------------------------------------------------------
def y0_from_logm500(
    log10M500,
    z,
    tckQFit,
    Ez_fn: interpolate.interp1d,
    tenToA0=4.95e-5,
    B0=0.08,
    Mpivot=3e14,
    sigma_int=0.2,
    fRelWeightsDict={148.0: 1.0},
    H0=70.0,
    Da_fn=None,
):
    """Predict y0~ given logM500 (in MSun) and redshift. Default scaling relation
    parameters are A10 (as in H13).

    Use cosmoModel (astropy.cosmology object) to change/specify cosmological parameters.

    fRelWeightsDict is used to account for the relativistic correction when y0~ has been
    constructed from multi-frequency maps. Weights should sum to 1.0; keys are observed
    frequency in GHz.

    Returns y0~, theta500Arcmin, Q

    """

    if type(Mpivot) == str:
        raise Exception(
            "Mpivot is a string - check Mpivot in your .yml config file:\
             use, e.g., 3.0e+14 (not 3e14 or 3e+14)"
        )

    # Filtering/detection was performed with a fixed fiducial cosmology... so we don't
    # need to recalculate Q.
    # We just need to recalculate theta500Arcmin and E(z) only
    M500 = np.power(10, log10M500)
    theta500Arcmin = calcTheta500Arcmin(z, M500, Ez_fn, Da_fn, H0)
    Q = calcQ(theta500Arcmin, tckQFit)

    Ez = Ez_fn(z)

    # Relativistic correction: now a little more complicated, to account for fact y0~ maps
    # are weighted sum of individual frequency maps, and relativistic correction size
    # varies with frequency
    fRels = []
    freqWeights = []
    for obsFreqGHz in fRelWeightsDict.keys():
        fRels.append(calcFRel(z, M500, obsFreqGHz=obsFreqGHz, Ez_fn=Ez_fn))
        freqWeights.append(fRelWeightsDict[obsFreqGHz])
    fRel = np.average(np.array(fRels), axis=0, weights=freqWeights)

    # UPP relation according to H13
    # NOTE: m in H13 is M/Mpivot
    # NOTE: this goes negative for crazy masses where the Q polynomial fit goes -ve, so
    # ignore those
    y0pred = tenToA0 * np.power(Ez, 2) * np.power(M500 / Mpivot, 1 + B0) * Q * fRel

    return y0pred, theta500Arcmin, Q
