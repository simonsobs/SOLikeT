import numpy as np
from scipy.special import spence
from scipy.optimize import fmin
from scipy.optimize import newton

from ..constants import MSUN_CGS, G_CGS, C_M_S, MPC2CM

delx = 0.01
C_CGS = C_M_S * 1.0e2
kpc_cgs = MPC2CM * 1.0e-3


def nfw(x):
    """shape of a NFW profile (NFW 1997, ApJ,490, 493)"""
    ans = 1.0 / (x * (1 + x) ** 2)
    return ans


def gx(x):
    ans = np.log(1.0 + x) - x / (1.0 + x)
    return ans


def gc(c):
    ans = 1.0 / (np.log(1.0 + c) - c / (1.0 + c))
    return ans


def Hc(c):
    ans = (
        -1.0 * np.log(1 + c) / (1.0 + c) + c * (1.0 + 0.5 * c) / ((1.0 + c) ** 2)
    ) / gx(c)
    return ans


def Sc(c):
    ans = (
        0.5 * np.pi**2
        - np.log(c) / 2.0
        - 0.5 / c
        - 0.5 / (1 + c) ** 2
        - 3 / (1 + c)
        + np.log(1 + c) * (0.5 + 0.5 / c**2 - 2 / c - 1 / (1 + c))
        + 1.5 * (np.log(1 + c)) ** 2
        + 3.0 * spence(c + 1)
    )

    return ans


def del_s(c):
    ans = Sc(c) / (Sc(c) + (1.0 / c**3) * Hc(c) * gx(c))
    return ans


def K_c(c):
    ans = 1.0 / 3.0 * Hc(c) / (1.0 - del_s(c))
    return ans


def sig_dm2(x, c):
    """EQ 14 Lokas & Mamon 2001"""
    ans = (
        0.5
        * x
        * c
        * gc(c)
        * (1 + x) ** 2
        * (
            np.pi**2
            - np.log(x)
            - (1.0 / x)
            - (1.0 / (1.0 + x) ** 2)
            - (6.0 / (1.0 + x))
            + np.log(1.0 + x) * (1.0 + (1.0 / x**2) - 4.0 / x - 2 / (1 + x))
            + 3.0 * (np.log(1.0 + x)) ** 2
            + 6.0 * spence(x + 1)
        )
    )
    return ans


def r200(M, z, provider):
    """radius of a sphere with density 200 times the critical density of the universe.
    Input mass in solar masses. Output radius in cm.
    """
    M_cgs = M * MSUN_CGS
    om = provider.get_param("Omega_m")
    ol = provider.get_param("Omega_L")
    Ez2 = om * (1 + z) ** 3 + ol
    rhocrit = 1.87847e-29 * provider.get_param("hh") ** 2
    ans = (3 * M_cgs / (4 * np.pi * 200.0 * rhocrit * Ez2)) ** (1.0 / 3.0)
    return ans


def con(M, z):
    """
    concentration parameter from Duffy et al. (2008)
    input mass in solar masses
    """
    ans = 5.71 / (1 + z) ** (0.47) * (M / 2e12) ** (-0.084)
    return ans


def jx(x, c):
    ans = 1.0 - np.log(1.0 + x) / x
    ind = np.where(x > c)  # [0]
    if len(ind) > 0:
        ans[ind] = 1.0 - 1.0 / (1.0 + c) - (np.log(1.0 + c) - c / (1.0 + c)) / x[ind]
    return ans


def jx_f(x, c):
    if x <= c:
        ans = 1.0 - np.log(1.0 + x) / x
    else:
        ans = 1.0 - 1.0 / (1.0 + c) - (np.log(1.0 + c) - c / (1.0 + c)) / x
    return ans


def fx(x, c):
    ans = np.log(1.0 + x) / x - 1.0 / (1.0 + c)
    ind = np.where(x > c)[0]
    if len(ind) > 0:
        ans = (np.log(1.0 + c) / c - 1.0 / (1.0 + c)) * c / x
    return ans


def fstar_func(M):
    """Giodini 2009, modified by 0.5"""
    ans = 2.5e-2 * (M / (7e13)) ** (-0.37)
    return ans


def xs_min_func(x, M, z):
    c = con(M, z)
    fstar = fstar_func(M)
    ans = gx(c) * fstar / (1.0 + fstar) - gx(x)
    return ans


def xs_func(M, z):
    x0 = 1.0
    xs = newton(
        xs_min_func,
        x0,
        args=(
            M,
            z,
        ),
    )
    return xs


def Ks(x_s, M, z):
    c = con(M, z)
    xx = np.arange(delx / 2.0, x_s, delx)
    ans = (
        1.0
        / gx(c)
        * (
            np.sum(Sc(xx) * xx**2)
            - 2.0 / 3.0 * np.sum(fx(xx, c) * xx / (1.0 + xx) ** 2)
        )
        * delx
    )
    return ans


def n_exp(gamma):
    """exponent of the polytopic e.o.s."""
    ans = 1.0 / (gamma - 1)
    return ans


def theta_func(x, M, z, theta, theta2):
    """polytropic variable"""
    gamma, alpha, Ef = theta
    beta, x_f = theta2
    c = con(M, z)
    nn = n_exp(gamma)
    # print(jx(x,c).shape)
    ans = 1.0 - beta * jx(x, c) / (1.0 + nn)
    return ans


def theta_func_f(x, M, z, theta, theta2):
    gamma, alpha, Ef = theta
    beta, x_f = theta2
    c = con(M, z)
    nn = n_exp(gamma)
    ans = 1.0 - beta * jx_f(x, c) / (1.0 + nn)
    return ans


def rho_use(x, M, z, theta, theta2):
    gamma, alpha, Ef = theta
    beta, x_f = theta2
    nn = n_exp(gamma)
    ans = (theta_func(x, M, z, theta, theta2)) ** nn
    return ans


def rho1h_one_mass(x, M, z, theta, provider):
    gamma, alpha, Ef = theta
    theta2 = find_params_M(M, z, theta)
    P_0, rho_0, x_f = theta2
    nn = n_exp(gamma)
    c = con(M, z)
    rvir = r200(M, z, provider)
    M_cgs = M * MSUN_CGS
    beta = rho_0 / P_0 * G_CGS * M_cgs / rvir * c / gx(c)
    theta2_use = beta, x_f
    ans = (
        rho_0
        * (theta_func(x / (rvir / kpc_cgs / 1e3) * c, M, z, theta, theta2_use)) ** nn
    )
    return ans


def rho1h(x, M, z, theta, provider):
    gamma, alpha, Ef = theta
    nn = n_exp(gamma)
    # CMASS specific bins and weights
    b_cen = np.array([[12.27689266, 12.67884686, 13.16053855, 13.69871423]]).T
    p = np.array([4.13431979e-03, 1.31666601e-01, 3.36540698e-01, 8.13760167e-02])
    rh = []
    for i in range(0, len(b_cen)):
        m = 10 ** b_cen[i]
        thta2 = find_params_M(m, z, theta)
        P_0, rho_0, x_f = thta2
        c = con(m, z)
        rvir = r200(m, z, provider)
        M_cgs = m * MSUN_CGS
        beta = rho_0 / P_0 * G_CGS * M_cgs / rvir * c / gx(c)
        theta2_use = beta, x_f
        rh.append(
            rho_0
            * (theta_func(x / (rvir / kpc_cgs / 1e3) * c, M, z, theta, theta2_use))
            ** nn
        )
    rh = np.array(rh)
    rh_av = np.average(rh, weights=p, axis=0)
    return rh_av


def rho2h(xx, theta2h):
    # CMASS specific 2h
    rho_file = np.genfromtxt(
        "/home/cemoser/Repositories/SOLikeT/soliket/szlike/twohalo_cmass_average.txt"
    )
    x1 = rho_file[:, 0]
    rho2h = rho_file[:, 1]
    ans = np.interp(xx, x1, rho2h)
    return theta2h * ans


def rho(x, M, z, theta, provider):
    theta1h = theta[0], theta[1], theta[2]
    theta2h = theta[3]
    ans = rho1h_one_mass(x, M, z, theta1h, provider) + rho2h(x, theta2h)
    return ans


def rho_outtest(x, M, z, theta, theta2, provider):
    gamma, alpha, Ef = theta
    P_0, rho_0, x_f = theta2
    nn = n_exp(gamma)
    c = con(M, z)
    rvir = r200(M, z, provider)
    M_cgs = M * MSUN_CGS
    beta = rho_0 / P_0 * G_CGS * M_cgs / rvir * c / gx(c)
    theta2_use = beta, x_f
    ans = rho_0 * (theta_func(x, M, z, theta, theta2_use)) ** nn
    return ans


def Pnth_th(x, M, z, theta):
    gamma, alpha, Ef = theta
    c = con(M, z)
    ans = 1.0 - alpha * (x / c) ** 0.8
    return ans


def Pth1h_one_mass(x, M, z, theta, provider):
    gamma, alpha, Ef = theta
    theta2 = find_params_M(M, z, theta)
    P_0, rho_0, x_f = theta2
    c = con(M, z)
    rvir = r200(M, z, provider)
    nn = n_exp(gamma)
    M_cgs = M * MSUN_CGS
    beta = rho_0 / P_0 * G_CGS * M_cgs / rvir * c / gx(c)
    theta2_use = beta, x_f
    ans = (
        P_0
        * (theta_func(x, M, z, theta, theta2_use)) ** (nn + 1.0)
        * Pnth_th(x, M, z, theta)
    )
    return ans


def Pth1h(x, M, z, theta, provider):
    gamma, alpha, Ef = theta
    nn = n_exp(gamma)
    # CMASS specific bins and weights
    b_cen = np.array(
        [
            [
                11.31932504,
                11.43785913,
                11.57526319,
                11.74539764,
                11.97016907,
                12.27689266,
                12.67884686,
                13.16053855,
                13.69871423,
            ]
        ]
    ).T
    p = np.array(
        [
            2.94467222e-06,
            2.94467222e-06,
            2.94467222e-06,
            1.47233611e-05,
            3.38637305e-05,
            4.13431979e-03,
            1.31666601e-01,
            3.36540698e-01,
            8.13760167e-02,
        ]
    )
    pth = []
    for i in range(0, len(b_cen)):
        m = 10 ** b_cen[i]
        thta2 = find_params_M(m, z, theta)
        P_0, rho_0, x_f = thta2
        c = con(m, z)
        rvir = r200(m, z, provider)
        M_cgs = m * MSUN_CGS
        beta = rho_0 / P_0 * G_CGS * M_cgs / rvir * c / gx(c)
        theta2_use = beta, x_f
        pth.append(
            P_0
            * (theta_func(x / (rvir / kpc_cgs / 1e3) * c, m, z, theta, theta2_use))
            ** (nn + 1.0)
            * Pnth_th(x / (rvir / kpc_cgs / 1e3) * c, m, z, theta)
        )
    pth = np.array(pth)
    pth_av = np.average(pth, weights=p, axis=0)
    return pth_av


def Pth2h(xx, theta2h):
    # CMASS specific 2h file
    pth_file = np.genfromtxt(
        "/home/cemoser/Repositories/SOLikeT/soliket/szlike/twohalo_cmass_average.txt"
    )
    x1 = pth_file[:, 0]
    pth2h = pth_file[:, 2]
    ans = np.interp(xx, x1, pth2h)
    return theta2h * ans


def Pth(x, M, z, theta):
    theta1h = theta[0], theta[1], theta[2]
    theta2h = theta[3]
    ans = Pth1h(x, M, z, theta1h, provider) + Pth2h(x, theta2h)
    return ans


def Pth_use(x, M, z, theta, theta2):
    gamma, alpha, Ef = theta
    beta, x_f = theta2
    nn = n_exp(gamma)
    ans = (theta_func(x, M, z, theta, theta2)) ** (nn + 1.0) * Pnth_th(x, M, z, theta)
    return ans


def Ptot(M, z, theta, theta2, provider):
    gamma, alpha, Ef = theta
    P_0, rho_0, x_f = theta2
    nn = n_exp(gamma)
    rvir = r200(M, z, provider)
    c = con(M, z)
    M_cgs = M * MSUN_CGS
    beta = rho_0 / P_0 * G_CGS * M_cgs / rvir * c / gx(c)
    theta2_use = beta, x_f
    ans = P_0 * (theta_func_f(x_f, M, z, theta, theta2_use)) ** (nn + 1.0)
    return ans


def Ptot_use(M, z, theta, theta2):
    gamma, alpha, Ef = theta
    beta, x_f = theta2
    nn = n_exp(gamma)
    ans = (theta_func_f(x_f, M, z, theta, theta2)) ** (nn + 1.0)
    return ans


def Pnth(x, M, z, theta, theta2, provider):
    gamma, alpha, Ef = theta
    P_0, rho_0, x_f = theta2
    c = con(M, z)
    nn = n_exp(gamma)
    rvir = r200(M, z, provider)
    M_cgs = M * MSUN_CGS
    beta = rho_0 / P_0 * G_CGS * Mvir / rvir * c / gx(c)
    theta2_use = beta, x_f
    ans = (
        alpha
        * (x / c) ** 0.8
        * P_0
        * (theta_func(x, M, z, theta, theta2_use)) ** (nn + 1.0)
    )
    return ans


def Pnth_use(x, M, z, theta, theta2):
    gamma, alpha, Ef = theta
    c = con(M, z)
    nn = n_exp(gamma)
    ans = alpha * (x / c) ** 0.8 * (theta_func(x, M, z, theta, theta2)) ** (nn + 1.0)
    return ans


def I2_int(M, z, theta, theta2):
    gamma, alpha, Ef = theta
    beta, x_f = theta2
    nn = n_exp(gamma)
    c = con(M, z)
    xx = np.arange(delx / 2.0, x_f, delx)
    ans = np.sum(fx(xx, c) * rho_use(xx, M, z, theta, theta2) * xx**2) * delx
    return ans


def I3_int(M, z, theta, theta2):
    gamma, alpha, Ef = theta
    beta, x_f = theta2
    nn = n_exp(gamma)
    xx = np.arange(delx / 2.0, x_f, delx)
    ans = np.sum(Pth_use(xx, M, z, theta, theta2) * xx**2) * delx
    return ans


def I4_int(M, z, theta, theta2):
    gamma, alpha, Ef = theta
    beta, x_f = theta2
    nn = n_exp(gamma)
    xx = np.arange(delx / 2.0, x_f, delx)
    ans = np.sum(Pnth_use(xx, M, z, theta, theta2) * xx**2) * delx
    return ans


def L_int(M, z, theta, theta2):
    gamma, alpha, Ef = theta
    beta, x_f = theta2
    nn = n_exp(gamma)
    xx = np.arange(delx / 2.0, x_f, delx)
    ans = np.sum(rho_use(xx, M, z, theta, theta2) * xx**2) * delx
    return ans


def rho_0_func(theta0, theta2, provider):
    M, z, gamma, alpha, Ef = theta0
    theta = [gamma, alpha, Ef]
    c = con(M, z)
    rvir = r200(M, z, provider)
    fstar = fstar_func(M)
    M_cgs = M * MSUN_CGS
    fb = provider.get_param("Omega_b") / provider.get_param("Omega_m")
    ans = (
        M_cgs
        * (fb - fstar)
        / (4.0 * np.pi * L_int(M, z, theta, theta2) * (rvir / c) ** 3)
    )
    return ans


def P_0_func(theta0, theta2, rho_0, provider):
    M, z, gamma, alpha, Ef = theta0
    beta, x_f = theta2
    c = con(M, z)
    rvir = r200(M, z, provider)
    M_cgs = M * MSUN_CGS
    ans = rho_0 / beta * G_CGS * M_cgs / rvir * c / gx(c)
    return ans


def findroots2(theta2, theta0, provider):
    M, z, gamma, alpha, Ef = theta0
    theta = [gamma, alpha, Ef]
    beta, x_f = theta2
    c = con(M, z)
    rvir = r200(M, z, provider)
    x_s = xs_func(M, z)
    fstar = fstar_func(M)
    M_cgs = M * MSUN_CGS

    E_inj = Ef * gx(c) * rvir * fstar / (G_CGS * M_cgs * c) * C_CGS**2

    Eq1 = (
        3.0 / 2.0 * (1.0 + fstar) * (K_c(c) * (3.0 - 4.0 * del_s(c)) + Ks(x_s, M, z))
        - E_inj
        + 1.0 / 3.0 * (1.0 + fstar) * Sc(c) / gx(c) * (x_f**3 - c**3)
        - I2_int(M, z, theta, theta2) / L_int(M, z, theta, theta2)
        + 3.0 / 2.0 * I3_int(M, z, theta, theta2) / (beta * L_int(M, z, theta, theta2))
        + 3.0 * I4_int(M, z, theta, theta2) / (beta * L_int(M, z, theta, theta2))
    )

    Eq2 = (1.0 + fstar) * Sc(c) / gx(c) * (
        beta * L_int(M, z, theta, theta2)
    ) - Ptot_use(M, z, theta, theta2)

    ans = Eq1**2 + Eq2**2
    return ans


def return_prof_pars(theta2, theta0):
    M, z, gamma, alpha, Ef = theta0
    beta, x_f = theta2
    ans = fmin(findroots2, theta2, args=(theta0,), disp=False)
    beta_ans, x_f_ans = ans
    rho_0 = rho_0_func(theta0, ans)
    P_0 = P_0_func(theta0, ans, rho_0)
    return P_0, rho_0, x_f_ans


def findroots(theta2, theta0, provider):
    M, z, gamma, alpha, Ef = theta0
    theta = [gamma, alpha, Ef]
    beta, x_f = theta2
    c = con(M, z)
    rvir = r200(M, z, provider)
    M_cgs = M * MSUN_CGS

    E_inj = Ef * gx(c) * rvir / (G_CGS * M_cgs * c) * C_CGS**2

    Eq1 = (
        3.0 / 2.0 * (K_c(c) * (3.0 - 4.0 * del_s(c)))
        - E_inj
        + 1.0 / 3.0 * Sc(c) / gx(c) * (x_f**3 - c**3)
        - I2_int(M, z, theta, theta2) / L_int(M, z, theta, theta2)
        + 3.0 / 2.0 * I3_int(M, z, theta, theta2) / (beta * L_int(M, z, theta, theta2))
        + 3.0 * I4_int(M, z, theta, theta2) / (beta * L_int(M, z, theta, theta2))
    )
    Eq2 = Sc(c) / gx(c) * (beta * L_int(M, z, theta, theta2)) - Ptot_use(
        M, z, theta, theta2
    )
    return (Eq1, Eq2)


def find_params_M(M, z, theta_0):
    theta0 = np.append([M, z], [theta_0])
    beta_0 = 1.1
    con_test = con(M, z)
    theta2 = np.array([beta_0, con_test * 1.01])
    ans = return_prof_pars(theta2, theta0)
    return ans
