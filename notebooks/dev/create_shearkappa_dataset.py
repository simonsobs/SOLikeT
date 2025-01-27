import cobaya
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sacc

import pyccl as ccl

from astropy import units

from cobaya.yaml import yaml_load_file
from cobaya.install import install
from cobaya.model import get_model

info = yaml_load_file('run_shearkappa_fiducial.yaml')

# here we will adopt a nominal cosmology to populate the sacc file
# before replacing with the spectra we just computed within soliket
# we also use the nominal cosmology for constructing a Gaussian covmat
h = 0.677

cosmo = ccl.Cosmology(Omega_c=info['params']['omch2']['value']/(h*h),
                      Omega_b=info['params']['ombh2']['value']/(h*h),
                      h=h,
                      n_s=info['params']['ns']['value'],
                      A_s=1e-10*np.exp(info['params']['logA']['value']))

# construct binning
ell_max = 1900
n_ell = 6
delta_ell = ell_max // n_ell

ells = (np.arange(n_ell) + 0.5) * delta_ell

ells_win = np.arange(ell_max + 1)
nell_win = len(ells_win)
# wins = np.zeros([n_ell, len(ells_win)])

# win_norm = n_ell / ell_max

# for i in range(n_ell):
#     wins[i, i * delta_ell : (i + 1) * delta_ell] = 1.0 * win_norm

# Well = sacc.BandpowerWindow(ells_win, wins.T)

# s_win = sacc.Sacc.load_fits('../../../../act-x-des/desgamma-x-actkappa/data/UNBLINDED_ACTPlanck_tSZfree_ACTDR4-kappa_DESY3-gamma_data_simCov.fits')

Well = sacc.BandpowerWindow(ells_win, np.loadtxt('bpw1.txt'))

# set up kappa lensing tracer
zstar = 1086

tracer_so_k = ccl.CMBLensingTracer(cosmo, z_source=zstar)
noise_so_kk = np.loadtxt('./data/nlkk_v3_1_0_deproj0_SENS1_fsky0p4_it_lT30-3000_lP30-5000.dat')[:,7] # [7 is MV(all)]
noise_so_kk = noise_so_kk[:ell_max+1]

# Approximation to SO LAT beam
fwhm_so_k = 1. * units.arcmin
sigma_so_k = (fwhm_so_k.to(units.rad).value / 2.355)
ell_beam = np.arange(3000)
beam_so_k = np.exp(-ell_beam * (ell_beam + 1) * sigma_so_k**2)

# set up shear lensing tracer
z_shear = np.loadtxt('./data/shearkappa_nz_source/z.txt')
nbins = 4
n_maps = nbins + 1
ngal = [1.5, 1.5, 1.5, 1.5]
sigma_e = [0.28, 0.28, 0.28, 0.28]

shear_tracers = []
shear_nz = []

for ibin in np.arange(1, nbins+1):

    nz_bin = np.loadtxt(f'./data/shearkappa_nz_source/bin_{ibin}.txt')
    shear_nz.append(nz_bin)
    z0_IA = np.trapz(z_shear * nz_bin)

    ia_z = (z_shear, info['params']['A_IA']['value'] * ((1 + z_shear) / (1 + z0_IA)) ** info['params']['eta_IA']['value'])

    tracer_bin = ccl.WeakLensingTracer(cosmo,
                                       dndz=(z_shear, nz_bin),
                                       ia_bias=ia_z)

    shear_tracers.append(tracer_bin)

# calculate spectra
spectra = np.zeros([n_maps, n_maps, nell_win])
spectra_label = np.empty([n_maps, n_maps], dtype='S2')
spectra[0, 0, :] = ccl.angular_cl(cosmo, tracer_so_k, tracer_so_k, ells_win) + noise_so_kk
spectra_label[0, 0] = 'kk'

for ibin in np.arange(1, nbins+1):

    Nell_bin = np.ones(nell_win) * sigma_e[ibin-1]**2. / (ngal[ibin-1] * (60 * 180 / np.pi)**2)

    spectra[ibin, 0, :] = ccl.angular_cl(cosmo, shear_tracers[ibin-1], tracer_so_k, ells_win)
    spectra[0, ibin, :] = spectra[ibin, 0, :]

    spectra_label[ibin, 0] = f'g{ibin}k'
    spectra_label[0, ibin] = f'kg{ibin}'

    for jbin in np.arange(1,nbins+1):
        if ibin==jbin:
            spectra[ibin, ibin, :] = ccl.angular_cl(cosmo, shear_tracers[ibin-1], shear_tracers[ibin-1], ells_win) + Nell_bin
            spectra_label[ibin, ibin] = f'g{ibin}g{ibin}'
        else:
            spectra[ibin, jbin, :] = ccl.angular_cl(cosmo, shear_tracers[ibin-1], shear_tracers[jbin-1], ells_win)
            spectra[jbin, ibin, :] = ccl.angular_cl(cosmo, shear_tracers[jbin-1], shear_tracers[ibin-1], ells_win)

            spectra_label[ibin, jbin] = f'g{ibin}g{jbin}'
            spectra_label[jbin, ibin] = f'g{jbin}g{ibin}'

# calculate covmat
n_maps = nbins + 1
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
                cl_i1j1_label = spectra_label[i1, j1]
                cl_i1j2 = np.dot(Well.weight.T, spectra[i1, j2, :])
                cl_i1j2_label = spectra_label[i1, j2]
                cl_i2j1 = np.dot(Well.weight.T, spectra[i2, j1, :])
                cl_i2j1_label = spectra_label[i2, j1]
                cl_i2j2 = np.dot(Well.weight.T, spectra[i2, j2, :])
                cl_i2j2_label = spectra_label[i2, j2]
                # Knox formula
                cov = (cl_i1j1 * cl_i2j2 + cl_i1j2 * cl_i2j1) / (d_ell * fsky * (2 * ells + 1))
                covar[id_i, :, id_j, :] = np.diag(cov)
                print(f'cov({id_i}, {id_j}): ({cl_i1j1_label} * {cl_i2j2_label} + {cl_i1j2_label} * {cl_i2j1_label})')
                id_j += 1
        id_i += 1

covar = covar.reshape([n_cross * n_ell, n_cross * n_ell])
print(covar.shape)

# construct sacc file
s = sacc.Sacc()

s.add_tracer('Map', 'ck_so',
            quantity='cmb_convergence',
            spin=0,
            ell=ell_beam,
            beam=beam_so_k)

for ibin in np.arange(1,nbins+1):

    s.add_tracer('NZ', 'gs_des_bin{}'.format(ibin),
                quantity='galaxy_shear',
                spin=2,
                z=z_shear,
                nz=shear_nz[ibin-1],
                metadata={
                          'sigma_e' : sigma_e[ibin-1],
                          'ngal' : ngal[ibin-1]
                         })

s.add_ell_cl('cl_00',
             'ck_so',
             'ck_so',
             ells, np.dot(Well.weight.T, spectra[0, 0, :]),
             window=Well)

for ibin in np.arange(1,nbins+1):
    s.add_ell_cl('cl_20',
                 'gs_des_bin{}'.format(ibin),
                 'ck_so',
                 ells, np.dot(Well.weight.T, spectra[ibin, 0, :]),
                 window=Well)

    for jbin in np.arange(1,nbins+1):
        if ibin<=jbin:
            s.add_ell_cl('cl_ee',
                         'gs_des_bin{}'.format(ibin),
                         'gs_des_bin{}'.format(jbin),
                         ells, np.dot(Well.weight.T, spectra[ibin, jbin, :]),
                         window=Well)



s.add_covariance(covar)

# we only want the cross-correlations in the sacc we're going to save:
keep_spectra = (('gs_des_bin1', 'ck_so'), ('gs_des_bin2', 'ck_so'), ('gs_des_bin3', 'ck_so'), ('gs_des_bin4', 'ck_so'))

for tracer_comb in s.get_tracer_combinations():
    if tracer_comb not in keep_spectra:
        s.remove_selection(tracers=tracer_comb)

s.save_fits('./data/shearkappa_smooth_mockdata.fits', overwrite=True)

# now we calculate the soliket spectra at the fiducial parameters

fid_cosmo = {'H0': info['params']['H0']['value'],
             'logA': info['params']['logA']['value'],
             'omch2': info['params']['omch2']['value'],
             'A_IA': info['params']['A_IA']['value'],
             'eta_IA': info['params']['eta_IA']['value'],}

# force model computation at fiducial parameters
model = get_model(info)
model.loglikes(fid_cosmo)

# and replace the data in the sacc file
sklike = model.likelihood['soliket.cross_correlation.ShearKappaLikelihood']

# param_values = {'H0': 67.7, 'logA': 3.05, 'omch2': 0.1202, 'A_IA': 0.35, 'eta_IA': 1.66}
param_values = {}

for par in info['params']:
    try:
        param_values[par] = info['params'][par]['value']
    except KeyError:
        try:
            param_values[par] = info['params'][par]['value']['loc']
        except:
            try:
                param_values[par] = info['params'][par]['value']
            except:
                continue

sktheory = sklike._get_theory(**param_values)

s.mean = sktheory

s.save_fits('./data/shearkappa_smooth_mockdata.fits', overwrite=True)

model = get_model(info)
likes = model.loglikes(fid_cosmo)

sklike = model.likelihood['soliket.cross_correlation.ShearKappaLikelihood']
sktheory = sklike._get_theory(**param_values)

assert np.all(s.mean==sktheory)
