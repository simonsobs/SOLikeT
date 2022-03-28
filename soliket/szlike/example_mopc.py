import numpy as np
import matplotlib.pyplot as plt
import mopc as mop
from mopc.beam import f_beam_150


"""choose halo redshift and mass [Msun]
"""
z = 0.57
m = 2e13

"""choose radial range (x = r/rvir)
"""
x = np.logspace(np.log10(0.1), np.log10(10), 100)

"""choose angular range [eg. arcmin]
"""
theta = np.arange(100) * 0.05 + 0.5
sr2sqarcmin = 3282.8 * 60.0 ** 2

"""
choose observing frequency [GHz]
"""
nu = 150.0


##########################################################################################
"""project a gNFW density profile [cgs unit] from Battaglia 2016
into a T_kSZ profile [muK arcmin^2]
"""
rho0 = np.log10(4e3 * (m / 1e14) ** 0.29 * (1 + z) ** (-0.66))
xc = 0.5
bt = 3.83 * (m / 1e14) ** 0.04 * (1 + z) ** (-0.025)
par_rho = [rho0, xc, bt, 1]
rho_gnfw = mop.rho_gnfw1h(x, m, z, par_rho)

print("making temp_ksz_gnfw...")
temp_ksz_gnfw = mop.make_a_obs_profile_sim_rho(theta, m, z, par_rho, f_beam_150)
print("done.")


plt.figure(0, figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.loglog(x, rho_gnfw, lw=3, label="gNFW density")
plt.xlabel(r"x", size=14)
plt.ylabel(r"$\rho_{gas}(x) [g/cm^3]$ ", size=14)
# plt.legend()
plt.subplot(1, 2, 2)
plt.plot(theta, temp_ksz_gnfw * sr2sqarcmin, lw=3)
plt.xlabel(r"$\theta$ [arcmin]", size=14)
plt.ylabel(r"$T_{kSZ} [\mu K \cdot arcmin^2]$", size=14)
plt.tight_layout()
plt.savefig("rho_gnfw.pdf")

##########################################################################################
"""project a gNFW profile of the thermal Pressure [cgs unit] from Battaglia et al. 2012
into a T_tSZ profile [muK arcmin^2]
"""

P0 = 18.1 * (m / 1e14) ** 0.154 * (1 + z) ** (-0.758)
al = 1.0
bt = 4.35 * (m / 1e14) ** 0.0393 * (1 + z) ** 0.415
par_pth = [P0, al, bt, 1]
pth_gnfw = mop.Pth_gnfw1h(x, m, z, par_pth)
temp_tsz_gnfw = mop.make_a_obs_profile_sim_pth(theta, m, z, par_pth, nu, f_beam_150)

plt.figure(1, figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.loglog(x, pth_gnfw, lw=3, label="gNFW pressure")
plt.xlabel(r"x", size=14)
plt.ylabel(r"$P_{th}(x) [dyne/cm^2]$", size=14)
# plt.legend()
plt.subplot(1, 2, 2)
plt.plot(theta, temp_tsz_gnfw * sr2sqarcmin, lw=3)
plt.xlabel(r"$\theta$ [arcmin]", size=14)
plt.ylabel(r"$T_{tSZ} [\mu K \cdot arcmin^2]$", size=14)
plt.tight_layout()
plt.savefig("pth_gnfw.pdf")

# ##########################################################################################
# """project density and thermal pressure models [cgs unit] from Ostriker, Bode, Balbus (2005)
# into T_kSZ and T_tSZ profiles [muK arcmin^2]
# """
# gamma = 1.2  # polytropic index
# alpha_NT = 0.13  # non-thermal pressure norm.
# eff = 2e-5  # feedback efficiency
# par_obb = [gamma, alpha_NT, eff, 1, 1]
# par2_obb = mop.find_params_M(m, z, par_obb)  # P_0, rho_0, x_f
# rho_obb = mop.rho(x, m, z, par_obb, par2_obb)
# pth_obb = mop.Pth(x, m, z, par_obb, par2_obb)
# temp_ksz_obb = mop.make_a_obs_profile(theta, m, z, par_obb, nu, f_beam_150)[0]
# temp_tsz_obb = mop.make_a_obs_profile(theta, m, z, par_obb, nu, f_beam_150)[1]

# plt.figure(2, figsize=(10, 4))
# plt.subplot(1, 2, 1)
# plt.loglog(x, rho_obb, lw=3, label="OBB density")
# plt.xlabel(r"x", size=14)
# plt.ylabel(r"$\rho_{gas}(x) [g/cm^3]$ ", size=14)
# # plt.legend()
# plt.subplot(1, 2, 2)
# plt.plot(theta, temp_ksz_obb * sr2sqarcmin, lw=3)
# plt.xlabel(r"$\theta$ [arcmin]", size=14)
# plt.ylabel(r"$T_{kSZ} [\mu K \cdot arcmin^2]$", size=14)
# plt.tight_layout()
# plt.savefig("/fig/rho_obb.pdf")

# plt.figure(3, figsize=(10, 4))
# plt.subplot(1, 2, 1)
# plt.loglog(x, pth_obb, lw=3, label="OBB pressure")
# plt.xlabel(r"x", size=14)
# plt.ylabel(r"$P_{th}(x) [dyne/cm^2]$", size=14)
# # plt.legend()
# plt.subplot(1, 2, 2)
# plt.plot(theta, temp_tsz_obb * sr2sqarcmin, lw=3)
# plt.xlabel(r"$\theta$ [arcmin]", size=14)
# plt.ylabel(r"$T_{tSZ} [\mu K \cdot arcmin^2]$", size=14)
# plt.tight_layout()
# plt.savefig("/fig/pth_obb.pdf")
