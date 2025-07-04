import numpy as np
import pyccl as ccl
import sacc
from astropy import units
from cobaya.model import get_model
from cobaya.yaml import yaml_load_file

info = yaml_load_file("./yamls/run_shearshear_fiducial.yaml")

fiducial_cosmo = yaml_load_file("./yamls/params_cosmo_smooth_fiducial.yaml")
fiducial_sys = yaml_load_file("./yamls/params_shearshearnuisance_smooth_fiducial.yaml")

fiducial_params = {**fiducial_cosmo, **fiducial_sys}
info["params"] = fiducial_params

# here we will adopt a nominal cosmology to populate the sacc file
# before replacing with the spectra we just computed within soliket
# we also use the nominal cosmology for constructing a Gaussian covmat
h = 0.677

cosmo = ccl.Cosmology(
    Omega_c=info["params"]["omch2"]["value"] / (h * h),
    Omega_b=info["params"]["ombh2"]["value"] / (h * h),
    h=h,
    n_s=info["params"]["ns"]["value"],
    A_s=1e-10 * np.exp(info["params"]["logA"]["value"]),
)

# construct binning
ell_max = 1900
n_ell = 6
delta_ell = ell_max // n_ell

ells = (np.arange(n_ell) + 0.5) * delta_ell

ells_win = np.arange(ell_max + 1)
nell_win = len(ells_win)
wins = np.zeros([n_ell, len(ells_win)])

win_norm = n_ell / ell_max

for i in range(n_ell):
    wins[i, i * delta_ell : (i + 1) * delta_ell] = 1.0 * win_norm

Well = sacc.BandpowerWindow(ells_win, wins.T)

# dir = '../../../../act-x-des/desgamma-x-actkappa/data/'
# fname = dir + 'UNBLINDED_ACTPlanck_tSZfree_ACTDR4-kappa_DESY3-gamma_data_simCov.fits'
# s_win = sacc.Sacc.load_fits(fname)

# Well = sacc.BandpowerWindow(ells_win, np.loadtxt("bpw1.txt"))

# set up shear lensing des
z_shear = np.loadtxt("./data/shearkappa_nz_source/z.txt")
nbins = 4
n_maps = nbins + 1
ngal = [1.5, 1.5, 1.5, 1.5]
sigma_e = [0.28, 0.28, 0.28, 0.28]

tracers = []
tracer_labels = []
shear_nz = []

for ibin in np.arange(1, nbins + 1):
    nz_bin = np.loadtxt(f"./data/shearkappa_nz_source/bin_{ibin}.txt")
    shear_nz.append(nz_bin)
    z0_IA = np.trapz(z_shear * nz_bin)

    ia_z = (
        z_shear,
        info["params"]["A_IA"]["value"]
        * ((1 + z_shear) / (1 + z0_IA)) ** info["params"]["eta_IA"]["value"],
    )

    tracer_bin = ccl.WeakLensingTracer(cosmo, dndz=(z_shear, nz_bin), ia_bias=ia_z)

    tracers.append(tracer_bin)
    tracer_labels.append(f'd{ibin}')

# set up shear lensing ska
z_shear = np.loadtxt("./data/shearkappa_nz_source/z.txt")
nbins = 4
n_maps = nbins + 1
ngal = [1.5, 1.5, 1.5, 1.5] + [1.5, 1.5, 1.5, 1.5]
sigma_e = [0.28, 0.28, 0.28, 0.28] + [0.28, 0.28, 0.28, 0.28]

for ibin in np.arange(1, nbins + 1):
    # nz_bin = np.loadtxt(f"./data/shearshear_nz_ska1/bin_{ibin}.txt")
    nz_bin = np.loadtxt(f"./data/shearkappa_nz_source/bin_{ibin}.txt") # fictional for now
    z0_IA = np.trapz(z_shear * nz_bin)

    ia_z = (
        z_shear,
        info["params"]["A_IA"]["value"]
        * ((1 + z_shear) / (1 + z0_IA)) ** info["params"]["eta_IA"]["value"],
    )

    tracer_bin = ccl.WeakLensingTracer(cosmo, dndz=(z_shear, nz_bin), ia_bias=ia_z)

    tracers.append(tracer_bin)
    tracer_labels.append(f's{ibin}')

n_maps = len(tracers)

# calculate spectra
n_cross = (n_maps * (n_maps + 1)) // 2

spectra = np.zeros([n_maps, n_maps, nell_win])
spectra_label = np.empty([n_maps, n_maps], dtype="S4")

#####!!!! You have N maps. Form the auto and cross spectra of those maps and label them

for ibin in np.arange(0, n_maps):
    for jbin in np.arange(0, ibin+1):

        Nell_bin = (
        np.ones(nell_win)
        * sigma_e[ibin - 1] ** 2.0
        / (ngal[ibin - 1] * (1.08e4 / np.pi) ** 2)
        )

        # Nell_bin_oiv = (
        # np.ones(nell_win)
        # * sigma_e_oiv[ibin - 1] ** 2.0
        # / (ngal_oiv[ibin - 1] * (1.08e4 / np.pi) ** 2)
        # )
        

        if ibin == jbin:
            # auto spectra

            spectra[ibin, ibin, :] = ccl.angular_cl(cosmo, tracers[ibin], tracers[ibin], ells_win) + Nell_bin
            spectra_label[ibin, ibin] = f'{tracer_labels[ibin]}{tracer_labels[ibin]}'
            print(tracer_labels[ibin], tracer_labels[ibin])


            # t1_spectra[ibin, ibin, :] = ccl.angular_cl(cosmo, shear1_tracers[ibin], shear1_tracers[ibin], ells_win) + Nell_bin_ska
            # t2_spectra[ibin, ibin, :] = ccl.angular_cl(cosmo, shear2_tracers[ibin], shear2_tracers[ibin], ells_win) + Nell_bin_oiv
            # cross_spectra[ibin, ibin, :] = ccl.angular_cl(cosmo, shear1_tracers[ibin], shear2_tracers[ibin], ells_win)

            # print(ska_bins[ibin], ska_bins[ibin])
            # print(oiv_bins[ibin], oiv_bins[ibin])
            # print(ska_bins[ibin], oiv_bins[ibin])

            # t1_spectra_label[ibin, ibin] = f"s{ibin}s{ibin}"
            # t2_spectra_label[ibin, ibin] = f"o{ibin}o{ibin}"
            # cross_spectra_label[ibin, ibin] = f"s{ibin}o{ibin}"

        else:
            # cross spectra
            spectra[ibin, jbin, :] = ccl.angular_cl(cosmo, tracers[ibin], tracers[jbin], ells_win)
            spectra[jbin, ibin, :] = ccl.angular_cl(cosmo, tracers[ibin], tracers[jbin], ells_win)

            spectra_label[ibin, jbin] = f'{tracer_labels[ibin]}{tracer_labels[jbin]}'
            spectra_label[jbin, ibin] = f'{tracer_labels[jbin]}{tracer_labels[ibin]}'
            print(tracer_labels[ibin], tracer_labels[jbin])

            # cross tomo-bin spectra
            # t1_spectra[ibin, jbin, :] = ccl.angular_cl(cosmo, shear1_tracers[ibin], shear1_tracers[jbin], ells_win)
            # t2_spectra[ibin, jbin, :] = ccl.angular_cl(cosmo, shear2_tracers[ibin], shear2_tracers[jbin], ells_win)
            # cross_spectra[ibin, jbin, :] = ccl.angular_cl(cosmo, shear1_tracers[ibin], shear2_tracers[jbin], ells_win)

            # t1_spectra_label[ibin, jbin] = f"s{ibin}s{ibin}"
            # t2_spectra_label[ibin, jbin] = f"o{ibin}o{jbin}"
            # cross_spectra_label[ibin, jbin] = f"s{ibin}o{jbin}"

            # print(ska_bins[ibin], ska_bins[jbin])
            # print(oiv_bins[ibin], oiv_bins[jbin])
            # print(ska_bins[ibin], oiv_bins[jbin])

# for ibin in np.arange(0, nbins + 1):
#     Nell_bin = (
#         np.ones(nell_win)
#         * sigma_e[ibin - 1] ** 2.0
#         / (ngal[ibin - 1] * (1.08e4 / np.pi) ** 2)
#     )

#     # spectra[ibin, 0, :] = ccl.angular_cl(
#     #     cosmo, shear1_tracers[ibin - 1], tracer_so_k, ells_win
#     # )
#     # spectra[0, ibin, :] = spectra[ibin, 0, :]

#     # spectra_label[ibin, 0] = f"g{ibin}k"
#     # spectra_label[0, ibin] = f"kg{ibin}"

#     for jbin in np.arange(0, nbins + 1):
#         if ibin == jbin:
#             spectra[ibin, ibin, :] = (
#                 ccl.angular_cl(
#                     cosmo, shear_tracers[ibin - 1], shear_tracers[ibin - 1], ells_win
#                 )
#                 + Nell_bin
#             )
#             spectra_label[ibin, ibin] = f"g{ibin}g{ibin}"
#         else:
#             spectra[ibin, jbin, :] = ccl.angular_cl(
#                 cosmo, shear_tracers[ibin - 1], shear_tracers[jbin - 1], ells_win
#             )
#             spectra[jbin, ibin, :] = ccl.angular_cl(
#                 cosmo, shear_tracers[jbin - 1], shear_tracers[ibin - 1], ells_win
#             )

#             spectra_label[ibin, jbin] = f"g{ibin}g{jbin}"
#             spectra_label[jbin, ibin] = f"g{jbin}g{ibin}"

# import pdb; pdb.set_trace()

# calculate covmat
# n_maps = nbins
fsky = 0.4
n_cross = (n_maps * (n_maps + 1)) // 2
covar = np.zeros([n_cross, n_ell, n_cross, n_ell])
d_ell = 30

id_i = 0
for i1 in range(n_maps):
    for i2 in range(i1, n_maps):
        id_j = 0
        for j1 in range(n_maps):
            for j2 in range(j1, n_maps):
                cl_i1j1 = np.dot(Well.weight.T, spectra[i1, j1, :])
                cl_i1j1_lab = spectra_label[i1, j1]
                cl_i1j2 = np.dot(Well.weight.T, spectra[i1, j2, :])
                cl_i1j2_lab = spectra_label[i1, j2]
                cl_i2j1 = np.dot(Well.weight.T, spectra[i2, j1, :])
                cl_i2j1_lab = spectra_label[i2, j1]
                cl_i2j2 = np.dot(Well.weight.T, spectra[i2, j2, :])
                cl_i2j2_lab = spectra_label[i2, j2]
                # Knox formula
                cov = (cl_i1j1 * cl_i2j2 + cl_i1j2 * cl_i2j1) / (
                    d_ell * fsky * (2 * ells + 1)
                )
                covar[id_i, :, id_j, :] = np.diag(cov)
                print(
                    f"cov({id_i}, {id_j}): ({cl_i1j1_lab} * {cl_i2j2_lab} + \
                      {cl_i1j2_lab} * {cl_i2j1_lab})"
                )
                id_j += 1
        id_i += 1

covar = covar.reshape([n_cross * n_ell, n_cross * n_ell])
print(covar.shape)

# construct sacc file
s = sacc.Sacc()

# s.add_tracer(
#     "Map", "ck_so", quantity="cmb_convergence", spin=0, ell=ell_beam, beam=beam_so_k
# )

for ibin in np.arange(1, nbins + 1):
    s.add_tracer(
        "NZ",
        "gs_{}".format(tracer_labels[ibin - 1]),
        quantity="galaxy_shear",
        spin=2,
        z=z_shear,
        nz=shear_nz[ibin - 1],
        metadata={"sigma_e": sigma_e[ibin - 1], "ngal": ngal[ibin - 1]},
    )

    s.add_tracer(
        "NZ",
        "gs_{}".format(tracer_labels[ibin + 3]),
        quantity="galaxy_shear",
        spin=2,
        z=z_shear,
        nz=shear_nz[ibin - 1],
        metadata={"sigma_e": sigma_e[ibin - 1], "ngal": ngal[ibin - 1]},
    )

# s.add_ell_cl(
#     "cl_00", "ck_so", "ck_so", ells, np.dot(Well.weight.T, spectra[0, 0, :]), window=Well
# )

for ibin in np.arange(1, n_maps + 1):
    # s.add_ell_cl(
    #     "cl_ee",
    #     "ss_des_bin{}".format(ibin),
    #     "ck_so",
    #     ells,
    #     np.dot(Well.weight.T, spectra[ibin, 0, :]),
    #     window=Well,
    # )

    for jbin in np.arange(1, n_maps + 1):
        if ibin <= jbin:
            s.add_ell_cl(
                "cl_ee",
                "gs_{}".format(tracer_labels[ibin - 1]),
                "gs_{}".format(tracer_labels[jbin - 1]),
                ells,
                np.dot(Well.weight.T, spectra[ibin - 1, jbin - 1, :]),
                window=Well,
            )


s.add_covariance(covar)

# we only want the cross-correlations in the sacc we're going to save:
keep_spectra = s.get_tracer_combinations()

for tracer_comb in s.get_tracer_combinations():
    if tracer_comb not in keep_spectra:
        s.remove_selection(tracers=tracer_comb)

s.save_fits("./data/shearshear_smooth_mockdata.fits", overwrite=True)

# # now we calculate the soliket spectra at the fiducial parameters

fid_cosmo = {
    "H0": info["params"]["H0"]["value"],
    "logA": info["params"]["logA"]["value"],
    "omch2": info["params"]["omch2"]["value"],
    "A_IA": info["params"]["A_IA"]["value"],
    "eta_IA": info["params"]["eta_IA"]["value"],
}

# force model computation at fiducial parameters
model = get_model(info)
model.loglikes(fid_cosmo)

# and replace the data in the sacc file
sslike = model.likelihood["soliket.cross_correlation.ShearShearLikelihood"]

# param_values = {'H0': 67.7, 'logA': 3.05, 'omch2': 0.1202, 'A_IA': 0.35, 'eta_IA': 1.66}
param_values = {}

for par in info["params"]:
    try:
        param_values[par] = info["params"][par]["value"]
    except KeyError:
        try:
            param_values[par] = info["params"][par]["value"]["loc"]
        except:
            try:
                param_values[par] = info["params"][par]["value"]
            except:
                continue

sstheory = sslike._get_theory(**param_values)

s.mean = sstheory

s.save_fits("./data/shearkappa_smooth_mockdata.fits", overwrite=True)

model = get_model(info)
likes = model.loglikes(fid_cosmo)

sslike = model.likelihood["soliket.cross_correlation.ShearShearLikelihood"]
sstheory = sslike._get_theory(**param_values)

assert np.all(s.mean == sstheory)
