from soliket.binned_clusters.binned_poisson import BinnedPoissonLikelihood
from scipy import interpolate, integrate, special
from scipy.interpolate import interp1d
from typing import Optional
import numpy as np
import math as m
import time as t
import os, sys
import multiprocessing
import astropy.table as atpy
from astropy.io import fits
from functools import partial
import logging
import nemo as nm
import scipy.stats
import pyccl as ccl
from classy_sz import Class


np.set_printoptions(threshold=sys.maxsize)


from pkg_resources import resource_filename

pi = 3.1415926535897932384630
rhocrit0 = 2.7751973751261264e11 # [h2 msun Mpc-3]
c_ms = 3e8                       # [m s-1]
Mpc = 3.08568025e22              # [m]
G = 6.67300e-11                  # [m3 kg-1 s-2]
msun = 1.98892e30                # [kg]

MPIVOT_THETA = 3e14 # [Msun]

class BinnedClusterLikelihood(BinnedPoissonLikelihood):

    name = "BinnedCluster"

    data: dict = {}
    theorypred: dict = {}
    YM: dict = {}
    selfunc: dict = {}
    binning: dict = {}
    verbose: bool = False

    params = {"tenToA0":None, "B0":None, "C0":None, "scatter_sz":None, "bias_sz":None}

    def initialize(self):

        self.log = logging.getLogger('BinnedCluster')
        handler = logging.StreamHandler()
        self.log.addHandler(handler)
        self.log.propagate = False
        if self.verbose:
            self.log.setLevel(logging.INFO)
        else:
            self.log.setLevel(logging.ERROR)

        self.log.info('Initializing binned_clusters_test.py')

        # SNR cut
        self.qcut = self.selfunc['SNRcut']

        if self.selfunc['mode'] == 'single_tile':
            self.log.info('Running single tile.')
        elif self.selfunc['mode'] == 'full':
            self.log.info('Running full analysis. No downsampling.')
        elif self.selfunc['mode'] == 'downsample':
            assert self.selfunc['dwnsmpl_bins'] is not None, 'mode = downsample but no bin number given. Aborting.'
            self.log.info('Downsampling selection function inputs.')
        elif self.selfunc['mode'] == 'inpt_dwnsmpld':
            self.log.info('Running on pre-downsampled input.')

        if self.selfunc['mode'] == 'single_tile':
            self.log.info('Considering only single tile.')
            self.datafile = self.data['cat_file']
        else:
            self.log.info("Considering full map.")
            self.datafile = self.data['cat_file']

        dimension = self.theorypred['choose_dim']
        if dimension == '2D':
            self.log.info('2D likelihood as a function of redshift and signal-to-noise.')
        else:
            self.log.info('1D likelihood as a function of redshift.')

        # reading catalogue
        self.log.info('Reading data catalog.')
        self.data_directory = self.data['data_path']
        list = fits.open(os.path.join(self.data_directory, self.datafile))
        data = list[1].data
        zcat = data.field("redshift")
        qcat = data.field("fixed_SNR") #NB note that there are another SNR in the catalogue

        # SPT-style SNR bias correction
        debiasDOF = 2
        qcat = np.sqrt(np.power(qcat, 2) - debiasDOF)

        qcut = self.qcut

        Ncat = len(zcat)
        self.log.info('Total number of clusters in catalogue = {}.'.format(Ncat))
        self.log.info('SNR cut = {}.'.format(qcut))

        z = zcat[qcat >= qcut]
        snr = qcat[qcat >= qcut]

        Ncat = len(z)
        self.log.info('Number of clusters above the SNR cut = {}.'.format(Ncat))
        self.log.info('The highest redshift = {}'.format(z.max()))

        # redshift bins for N(z)
        zbins = np.arange(self.binning['z']['zmin'], self.binning['z']['zmax'] + self.binning['z']['dz'], self.binning['z']['dz'])
        zarr = 0.5*(zbins[:-1] + zbins[1:])
        self.zarr = zarr

        self.log.info("Number of redshift bins = {}.".format(len(zarr)))

        # mass bin
        self.lnmmin = np.log(self.binning['M']['Mmin'])
        self.lnmmax = np.log(self.binning['M']['Mmax'])
        self.dlnm = self.binning['M']['dlogM']
        self.marr = np.arange(self.lnmmin+(self.dlnm/2.), self.lnmmax, self.dlnm)
        # this is to be consist with szcounts.f90 - maybe switch to linspace?

        self.log.info('Number of mass bins for theory calculation {}.'.format(len(self.marr)))
        #TODO: I removed the bin where everything is larger than zmax - is this ok?
        delNcat, _ = np.histogram(z, bins=zbins)

        self.delNcat = zarr, delNcat

        # SNR binning (following szcounts.f90)
        logqmin = self.binning['q']['log10qmin']
        logqmax = self.binning['q']['log10qmax']
        dlogq = self.binning['q']['dlog10q']

        # TODO: I removed the bin where everything is larger than qmax - is this ok?
        Nq = int((logqmax - logqmin)/dlogq) + 1
        # qbins = 10**np.arange(logqmin, logqmax+dlogq, dlogq)
        # qarr = 0.5*(qbins[:1] + qbins[1:])

        # constant binning in log10
        qbins = np.arange(logqmin, logqmax+dlogq, dlogq)
        qarr = 10**(0.5*(qbins[:-1] + qbins[1:]))

        # print('qbins:',np.log10(qarr))

        if dimension == "2D":
            self.log.info('The lowest SNR = {}.'.format(snr.min()))
            self.log.info('The highest SNR = {}.'.format(snr.max()))
            self.log.info('Number of SNR bins = {}.'.format(Nq))
            self.log.info('Edges of SNR bins = {}.'.format(qbins))

        delN2Dcat, _, _ = np.histogram2d(z, snr, bins=[zbins, 10**qbins])

        self.Nq = Nq
        self.qarr = qarr
        # self.qbins = qbins
        self.qbins = 10**qbins
        self.dlogq = dlogq
        self.delN2Dcat = zarr, qarr, delN2Dcat
        # print(self.delN2Dcat)
        # exit()

        # print('zbin:',zarr)

        self.log.info('Loading files describing selection function.')
        self.log.info('Reading Q as a function of theta.')
        if self.selfunc['mode'] == 'single_tile':
            self.log.info('Reading Q function for single tile.')
            self.datafile_Q = self.data['Q_file']
            list = fits.open(os.path.join(self.data_directory, self.datafile_Q))
            data = list[1].data
            self.tt500 = data.field("theta500Arcmin")
            self.Q = data.field("PRIMARY")
            assert len(self.tt500) == len(self.Q)
            self.log.info("Number of Q functions = {}.".format(len(self.Q[0])))

        else:
            if self.selfunc['mode'] == 'inpt_dwnsmpld':
                self.log.info('Reading pre-downsampled Q function.')
                # for quick reading theta and Q data is saved first and just called
                self.datafile_Q = self.data['Q_file']
                Qfile = np.load(os.path.join(self.data_directory, self.datafile_Q))
                self.tt500 = Qfile['theta']
                self.allQ = Qfile['Q']
                assert len(self.tt500) == len(self.allQ[:,0])

            else:
                self.datafile_Q = self.data['Q_file']
                filename_Q, ext = os.path.splitext(self.datafile_Q)
                datafile_Q_dwsmpld = os.path.join(self.data_directory,
                            filename_Q + 'dwsmpld_nbins={}'.format(self.selfunc['dwnsmpl_bins']) + '.npz')

                if self.selfunc['mode'] == 'full' or (
                        self.selfunc['mode'] == 'downsample' and self.selfunc['save_dwsmpld'] is False)  or (
                        self.selfunc['mode'] == 'downsample' and self.selfunc['save_dwsmpld'] and not os.path.exists(datafile_Q_dwsmpld)):
                    self.log.info('Reading full Q function.')
                    tile_area = np.genfromtxt(os.path.join(self.data_directory, self.data['tile_file']), dtype=str)
                    tilename = tile_area[:, 0]
                    QFit = nm.signals.QFit(QFitFileName=os.path.join(self.data_directory, self.datafile_Q), tileNames=tilename)
                    Nt = len(tilename)
                    self.log.info("Number of tiles = {}.".format(Nt))

                    hdulist = fits.open(os.path.join(self.data_directory, self.datafile_Q))
                    data = hdulist[1].data
                    tt500 = data.field("theta500Arcmin")

                    # reading in all Q functions
                    allQ = np.zeros((len(tt500), Nt))
                    for i in range(Nt):
                        allQ[:, i] = QFit.getQ(tt500, tileName=tile_area[:, 0][i])
                    assert len(tt500) == len(allQ[:, 0])
                    self.tt500 = tt500
                    self.allQ = allQ
                else:
                    self.log.info('Reading in binned Q function from file.')
                    Qfile = np.load(datafile_Q_dwsmpld)
                    self.allQ = Qfile['Q_dwsmpld']
                    self.tt500 = Qfile['tt500']

        self.log.info('Reading RMS.')
        if self.selfunc['mode'] == 'single_tile':
            self.datafile_rms = self.data['rms_file']

            list = fits.open(os.path.join(self.data_directory, self.datafile_rms))
            data = list[1].data
            self.skyfracs = data.field("areaDeg2")*np.deg2rad(1.)**2
            self.noise = data.field("y0RMS")
            self.log.info("Number of sky patches = {}.".format(self.skyfracs.size))

        else:
            if self.selfunc['mode'] == 'inpt_dwnsmpld':
                # for convenience,
                # save a down sampled version of rms txt file and read it directly
                # this way is a lot faster
                # could recreate this file with different downsampling as well
                # tile name is replaced by consecutive number from now on
                self.log.info('Reading pre-downsampled RMS table.')
                self.datafile_rms = self.data['rms_file']
                file_rms = np.loadtxt(os.path.join(self.data_directory, self.datafile_rms))
                self.noise = file_rms[:,0]
                self.skyfracs = file_rms[:,1]
                self.tname = file_rms[:,2]
                self.log.info("Number of tiles = {}. ".format(len(np.unique(self.tname))))
                self.log.info("Number of sky patches = {}.".format(self.skyfracs.size))
            else:
                self.datafile_rms = self.data['rms_file']
                filename_rms, ext = os.path.splitext(self.datafile_rms)
                datafile_rms_dwsmpld = os.path.join(self.data_directory,
                        filename_rms + 'dwsmpld_nbins={}'.format(self.selfunc['dwnsmpl_bins']) + '.' + '.npz')
                if self.selfunc['mode'] == 'full' or (
                        self.selfunc['mode'] == 'downsample' and self.selfunc['save_dwsmpld'] is False)  or (
                        self.selfunc['mode'] == 'downsample' and self.selfunc['save_dwsmpld'] and not os.path.exists(datafile_rms_dwsmpld)):
                    self.log.info('Reading in full RMS table.')

                    list = fits.open(os.path.join(self.data_directory, self.datafile_rms))
                    file_rms = list[1].data

                    self.noise = file_rms['y0RMS']
                    self.skyfracs = file_rms['areaDeg2']*np.deg2rad(1.)**2
                    self.tname = file_rms['tileName']
                    self.log.info("Number of tiles = {}. ".format(len(np.unique(self.tname))))
                    self.log.info("Number of sky patches = {}.".format(self.skyfracs.size))
                else:
                    self.log.info('Reading in binned RMS table from file.')
                    rms = np.load(datafile_rms_dwsmpld)
                    self.noise = rms['noise']
                    self.skyfracs = rms['skyfracs']
                    self.log.info("Number of rms bins = {}.".format(self.skyfracs.size))

        if self.selfunc['mode'] == 'downsample':
            if self.selfunc['save_dwsmpld'] is False or (self.selfunc['save_dwsmpld'] and not os.path.exists(datafile_Q_dwsmpld)):
                self.log.info('Downsampling RMS and Q function using {} bins.'.format(self.selfunc['dwnsmpl_bins']))
                binned_stat = scipy.stats.binned_statistic(self.noise, self.skyfracs, statistic='sum',
                                                           bins=self.selfunc['dwnsmpl_bins'])
                binned_area = binned_stat[0]
                binned_rms_edges = binned_stat[1]

                bin_ind = np.digitize(self.noise, binned_rms_edges)
                tiledict = dict(zip(tilename, np.arange(tile_area[:, 0].shape[0])))

                Qdwnsmpld = np.zeros((self.allQ.shape[0], self.selfunc['dwnsmpl_bins']))

                for i in range(self.selfunc['dwnsmpl_bins']):
                    tempind = np.where(bin_ind == i + 1)[0]
                    if len(tempind) == 0:
                        self.log.info('Found empty bin.')
                        Qdwnsmpld[:, i] = np.zeros(self.allQ.shape[0])
                    else:
                        temparea = self.skyfracs[tempind]
                        temptiles = self.tname[tempind]
                        test = [tiledict[key] for key in temptiles]
                        Qdwnsmpld[:, i] = np.average(self.allQ[:, test], axis=1, weights=temparea)

                self.noise = 0.5*(binned_rms_edges[:-1] + binned_rms_edges[1:])
                self.skyfracs = binned_area
                self.allQ = Qdwnsmpld
                self.log.info("Number of downsampled sky patches = {}.".format(self.skyfracs.size))

                assert self.noise.shape[0] == self.skyfracs.shape[0] and self.noise.shape[0] == self.allQ.shape[1]

                if self.selfunc['save_dwsmpld']:
                    np.savez(datafile_Q_dwsmpld, Q_dwsmpld=Qdwnsmpld, tt500=self.tt500)
                    np.savez(datafile_rms_dwsmpld, noise=self.noise, skyfracs=self.skyfracs)

        elif self.selfunc['mode'] == 'full':
            tiledict = dict(zip(tilename, np.arange(tile_area[:, 0].shape[0])))
            self.tile_list = [tiledict[key]+1 for key in self.tname]

        if self.selfunc['average_Q']:
            self.Q = np.mean(self.allQ, axis=1)
            self.log.info("Number of Q functions = {}.".format(self.Q.ndim))
            self.log.info("Using one averaged Q function for optimisation")
        else:
            self.Q = self.allQ
            self.log.info("Number of Q functions = {}.".format(len(self.Q[0])))

        self.log.info('Entire survey area = {} deg2.'.format(self.skyfracs.sum()/(np.deg2rad(1.)**2.)))
        # exit(0)

        # finner binning for low redshift
        minz = zarr[0]
        maxz = zarr[-1]
        if minz < 0: minz = 0.0
        zi = minz

        # counting redshift bins
        Nzz = 0
        while zi <= maxz :
            zi = self._get_hres_z(zi)
            Nzz += 1

        Nzz += 1
        zi = minz
        zz = np.zeros(Nzz)
        for i in range(Nzz):
            zz[i] = zi
            zi = self._get_hres_z(zi)
        if zz[0] == 0. : zz[0] = 1e-4 # 1e-8 = steps_z(Nz) in f90
        self.zz = zz
        print(" Nz for higher resolution = ", len(zz))

        # self.theorypred['use_class_sz'] = self.theorypred['self.theorypred['use_class_sz']']
        M = Class()
        class_sz_params = {
        'output' : 'dndlnM',
        'mass function'  : 'M500',
        'concentration parameter' : 'B13',
        'has_selection_function' : 1,
        'experiment' : 1.,
        #'sky_area_deg2' : 599.353 # not used when completeness demanded
        'y_m_relation' : 1,
        'use_planck_binned_proba' : 0, #use diff of erfs
        'class_sz_verbose'  : 0,

        'M_min' : 1e12,
        'M_max' : 1e16,



        # 'N_ur' : 2.0328,
        # 'N_ncdm' : 1,
        # 'm_ncdm' : 0.06,
        # 'T_ncdm' : 0.71611,

        'non linear' : 'halofit',


        # scaling law parameter
        # Hilton et al 2020
        'A_ym'  : 4.35e-5,
        'B_ym'  : 0.08,
        'm_pivot_ym [Msun]' : 3e14,
        'sigmaM_ym' : 0.,

        'omega_b': 0.0226576,
        'omega_cdm': 0.1206864,
        'n_s': 0.965,
        'tau_reio': 0.055,
        'H0': 68.,

        # X ray mass bias (if applicable)
        'B' : 1.,




        # tabulation of mass function:
        'n_z_dndlnM' : 100,
        'n_m_dndlnM' : 100,

        # computation of mass function
        # (grid in mass and redshifts for sigma and dsigma)
        'ndim_masses' : 100,
        'ndim_redshifts' : 100,

        # pk setup for computation of sigma and dsigma
        'k_per_decade_class_sz' : 20.,
        'k_min_for_pk_class_sz' : 1e-3,
        'k_max_for_pk_class_sz' : 1e1,
        'P_k_max_h/Mpc' : 1e1,

        'SO_thetas_file' : '/Users/boris/Work/CLASS-SZ/SO-SZ/class_sz/sz_auxiliary_files/advact_dr5_thetas_300621_1bins.txt',
        'SO_skyfracs_file' : '/Users/boris/Work/CLASS-SZ/SO-SZ/class_sz/sz_auxiliary_files/advact_dr5_skyfracs_300621_1bins.txt',
        'SO_ylims_file' : '/Users/boris/Work/CLASS-SZ/SO-SZ/class_sz/sz_auxiliary_files/advact_dr5_ylims_300621_1bins.txt'
        }
        if self.theorypred['use_class_sz']:
            M.set(class_sz_params)
            M.compute()
            self.get_dndlnM_at_z_and_M =  np.vectorize(M.get_dndlnM_at_z_and_M)
            self.get_y_at_m_and_z=  np.vectorize(M.get_y_at_m_and_z)
            self.get_theta_at_m_and_z=  np.vectorize(M.get_theta_at_m_and_z)


            self.log.info("[class_sz] class_sz initialized with:")
            self.log.info('[class_sz] h : {}'.format(M.h()))
            self.log.info('[class_sz] sigma8 : {}'.format(M.sigma8()))
            self.log.info('[class_sz] Omega_m : {}'.format(M.Omega_m()))
            self.log.info('[class_sz] n_s : {}'.format(M.n_s()))
            self.log.info('[class_sz] test dndlnm : {}'.format(self.get_dndlnM_at_z_and_M(0.1,3e14)))
            self.log.info('[class_sz] test ym : {}'.format(self.get_y_at_m_and_z(3e14,0.1)))
            self.log.info('[class_sz] test theta : {}'.format(self.get_theta_at_m_and_z(3e14,0.1)))
        # if self.theorypred['MiraTitanHMFemulator']:
        #     print('using MiraTitanHMFemulator')

        super().initialize()

    def get_requirements(self):
        if self.theorypred['choose_theory'] == "camb":
            req = {"Hubble":  {"z": self.zz},
                   "angular_diameter_distance": {"z": self.zz},
                   "H0": None, #NB H0 is derived
                   "Pk_interpolator": {"z": np.linspace(0, 3., 140), # should be less than 150
                                       "k_max": 4.0,
                                       "nonlinear": False,
                                       "hubble_units": False, # CLASS doesn't like this
                                       "k_hunit": False, # CLASS doesn't like this
                                       "vars_pairs": [["delta_nonu", "delta_nonu"]]}}
        elif self.theorypred['choose_theory'] == "class":
            req = {"Hubble":  {"z": self.zz},
                   "angular_diameter_distance": {"z": self.zz},
                   "Pk_interpolator": {"z": np.linspace(0, 3., 100), # should be less than 110
                                       "k_max": 4.0,
                                       "nonlinear": False,
                                       "vars_pairs": [["delta_nonu", "delta_nonu"]]}}
        elif self.theorypred['choose_theory'] == 'CCL':
            req = {'CCL': {},
                    'nc_data': {},
                    'Hubble': {'z': self.zz},
                    'angular_diameter_distance': {'z': self.zz},
                    'Pk_interpolator': {},
                    'H0': None  #NB H0 is derived
                    }
        else:
            raise NotImplementedError('Only theory modules camb, class and CCL implemented so far.')
        return req

    def _get_data(self):
        return self.delNcat, self.delN2Dcat

    def _get_om(self):
        if self.theorypred['choose_theory'] == "camb":
            om = (self.theory.get_param("omch2") + self.theory.get_param("ombh2") +
                  self.theory.get_param("omnuh2"))/((self.theory.get_param("H0")/100.0)**2)
        elif self.theorypred['choose_theory'] == "class":
            om = (self.theory.get_param("omega_cdm") +
                  self.theory.get_param("omega_b"))/((self.theory.get_param("H0")/100.0)**2) # for CLASS
        return om

    def _get_Ez(self, z):
        return self.theory.get_Hubble(z)/self.theory.get_param("H0")

    def _get_DAz(self, z):
        return self.theory.get_angular_diameter_distance(z)

    def _get_hres_z(self, zi):
        # bins in redshifts are defined with higher resolution for low redshift
        hr = 0.2
        if zi < hr :
            dzi = 1e-2
        elif zi >= hr and zi <=1.:
            dzi = 5e-2
        else:
            dzi = 5e-2#self.binning['z']['dz']
        hres_z = zi + dzi
        return hres_z

    def _get_dndlnm(self, z, pk_intp, **params_values_dict):

        #TODO: Why is zarr not used?
        zarr = self.zarr
        marr = self.marr  # Mass in units of Msun/h

        if self.theorypred['massfunc_mode'] == 'internal':
            h = self.theory.get_param("H0")/100.0
            Ez = self._get_Ez(z)
            om = self._get_om()
            rhom0 = rhocrit0 * om

            # redshift bin for P(z,k)
            zpk = np.linspace(0, 3., 200)
            if zpk[0] == 0.:
                zpk[0] = 1e-5

            k = np.logspace(-4, np.log10(4), 200, endpoint=False)
            pks0 = pk_intp.P(zpk, k)

            def pks_zbins(newz):
                newp = np.zeros((len(newz),len(k)))
                for i in range(k.size):
                    tck = interpolate.splrep(zpk, pks0[:,i])
                    newp[:,i] = interpolate.splev(newz, tck)
                return newp

            # rebin
            pks = pks_zbins(z)

            pks *= h**3.
            kh = k/h

            def radius(M): # R in units of Mpc/h
                return (0.75*M/pi/rhom0)**(1./3.)

            def win(x):
                return 3.*(np.sin(x) - x*np.cos(x))/(x**3.)

            def win_prime(x):
                return 3.*np.sin(x)/(x**2.) - 9.*(np.sin(x) - x*np.cos(x))/(x**4.)

            def sigma_sq(R, k):
                integral = np.zeros((len(k), len(marr), len(z)))
                for i in range(k.size):
                    integral[i,:,:] = np.array((k[i]**2.)*pks[:,i]*(win(k[i]*R)**2.))
                return integrate.simps(integral, k, axis=0)/(2.*pi**2.)

            def sigma_sq_prime(R, k):
                # this is derivative of sigmaR squared
                # so 2 * sigmaR * dsigmaR/dR
                integral = np.zeros((len(k), len(marr), len(z)))
                for i in range(k.size):
                    integral[i,:,:] = np.array((k[i]**2.)*pks[:,i]*2.*k[i]*win(k[i]*R)*win_prime(k[i]*R))
                return integrate.simps(integral, k, axis=0)/(2.*pi**2.)

            def tinker(sgm, z):

                total = 9
                delta  = np.zeros(total)
                par_aa = np.zeros(total)
                par_a  = np.zeros(total)
                par_b  = np.zeros(total)
                par_c  = np.zeros(total)

                delta[0] = 200
                delta[1] = 300
                delta[2] = 400
                delta[3] = 600
                delta[4] = 800
                delta[5] = 1200
                delta[6] = 1600
                delta[7] = 2400
                delta[8] = 3200

                par_aa[0] = 0.186
                par_aa[1] = 0.200
                par_aa[2] = 0.212
                par_aa[3] = 0.218
                par_aa[4] = 0.248
                par_aa[5] = 0.255
                par_aa[6] = 0.260
                par_aa[7] = 0.260
                par_aa[8] = 0.260

                par_a[0] = 1.47
                par_a[1] = 1.52
                par_a[2] = 1.56
                par_a[3] = 1.61
                par_a[4] = 1.87
                par_a[5] = 2.13
                par_a[6] = 2.30
                par_a[7] = 2.53
                par_a[8] = 2.66

                par_b[0] = 2.57
                par_b[1] = 2.25
                par_b[2] = 2.05
                par_b[3] = 1.87
                par_b[4] = 1.59
                par_b[5] = 1.51
                par_b[6] = 1.46
                par_b[7] = 1.44
                par_b[8] = 1.41

                par_c[0] = 1.19
                par_c[1] = 1.27
                par_c[2] = 1.34
                par_c[3] = 1.45
                par_c[4] = 1.58
                par_c[5] = 1.80
                par_c[6] = 1.97
                par_c[7] = 2.24
                par_c[8] = 2.44

                delta = np.log10(delta)
                omz = om*((1. + z)**3.)/(Ez**2.)

                if self.theorypred['md_hmf'] == '500c':
                    dsoz = 500./omz   # M500c
                elif self.theorypred['md_hmf'] == '200m':
                    dsoz = 200     # M200m
                else:
                    raise NotImplementedError()

                tck1 = interpolate.splrep(delta, par_aa)
                tck2 = interpolate.splrep(delta, par_a)
                tck3 = interpolate.splrep(delta, par_b)
                tck4 = interpolate.splrep(delta, par_c)

                par1 = interpolate.splev(np.log10(dsoz), tck1)
                par2 = interpolate.splev(np.log10(dsoz), tck2)
                par3 = interpolate.splev(np.log10(dsoz), tck3)
                par4 = interpolate.splev(np.log10(dsoz), tck4)

                alpha = 10.**(-((0.75/np.log10(dsoz/75.))**1.2))
                A     = par1*((1. + z)**(-0.14))
                a     = par2*((1. + z)**(-0.06))
                b     = par3*((1. + z)**(-alpha))
                c     = par4*np.ones(z.size)

                return A * (1. + (sgm/b)**(-a)) * np.exp(-c/(sgm**2.))

            dRdM = radius(np.exp(marr)) / (3. * np.exp(marr))
            dRdM = dRdM[:, None]
            R = radius(np.exp(marr))[:, None]
            sigma = sigma_sq(R, kh) ** 0.5
            sigma_prime = sigma_sq_prime(R, kh)
            hmf_internal = -rhom0 * tinker(sigma, z) * dRdM * (sigma_prime / (2. * sigma ** 2.))
            # print('goal shape:',np.shape(hmf_internal))
            # return -rhom0 * tinker(sigma, z) * dRdM * (sigma_prime / (2. * sigma ** 2.))
            # print(np.shape(marr),marr)
            # print(np.shape(z))
            if self.theorypred['use_class_sz']:
                hmf_class_sz = []
                for zz in z:
                    hmf_class_sz.append(self.get_dndlnM_at_z_and_M(zz,np.exp(marr)))
                # self.get_dndlnM_at_z_and_M(np.linspace(0.1,1.,10),marr)
                hmf_class_sz = np.asarray(hmf_class_sz)
                # print('class_sz shape:',np.shape(hmf_class_sz.T))

                return hmf_class_sz.T
            else:
                return -rhom0 * tinker(sigma, z) * dRdM * (sigma_prime / (2. * sigma ** 2.))

        elif self.theorypred['massfunc_mode'] == 'ccl':
            # First, gather all the necessary ingredients for the number counts
            mf = self.theory.get_nc_data()['HMF']
            cosmo = self.theory.get_CCL()['cosmo']

            h = self.theory.get_param("H0") / 100.0
            a = 1./(1+z)
            marr = np.exp(marr)
            dn_dlog10M = np.array([mf.get_mass_function(cosmo, marr/h, ai) for ai in a])
            # For consistency with internal mass function computation
            dn_dlog10M /= h**3*np.log(10.)

            return dn_dlog10M.T

        # elif self.theorypred['massfunc_mode'] == 'class_sz':
            # return self.get_dndlnM_at_z_and_M(z,marr)


    def _get_dVdzdO(self, z):

        dAz = self._get_DAz(z)
        Hz = self.theory.get_Hubble(z)
        h = self.theory.get_param("H0") / 100.0
        dVdzdO = (c_ms/1e3)*(((1. + z)*dAz)**2.)/Hz
        return dVdzdO * h**3.

    def _get_M500c_from_M200m(self, M200m, z):

        H0 = self.theory.get_param("H0")
        om = self._get_om()

        def Ehz(zz):
            return np.sqrt(om * np.power(1. + zz, 3.) + (1. - om))

        def growth(zz):
            zmax = 1000
            dz = 0.1
            zs = np.arange(zz, zmax, dz)
            y = (1 + zs)/ np.power(H0 * Ehz(zs), 3)
            return Ehz(zz) * integrate.simps(y, zs)

        def normalised_growth(zz):
            return growth(zz)/growth(0.)

        def rho_crit(zz):
            #GG = 4.301e-9  # in MSun-1 km2 s-2 Mpc ?
            return 3. / (8. * np.pi * G) * np.power(H0 * Ehz(zz), 2.)

        def rho_mean(zz):
            z0 = 0.
            return rho_crit(z0) * om * np.power(1 + zz, 3.)

        Dz = []
        for i in range(len(z)):
            Dz.append(normalised_growth(z[i]))
        Dz = np.array(Dz)

        rho_c = rho_crit(z)
        rho_m = rho_mean(z)
        M200m = M200m[:,None]

        #peak = (1. / Dz) * (1.12 * np.power(M200m / (5e13 / h), 0.3) + 0.53)
        peak = (1. / Dz) * (1.12 * np.power(M200m / 5e13, 0.3) + 0.53)
        c200m = np.power(Dz, 1.15) * 9. * np.power(peak, -0.29)
        R200m = np.power(3./(4. * np.pi) * M200m / (200. * rho_m), 1./3.)
        rs = R200m / c200m

        x = np.linspace(1e-3, 10, 1000)
        fx = np.power(x, 3.) * (np.log(1. + 1./x) - 1./(1. + x))

        xf_intp = interpolate.splrep(fx, x)
        fx_intp = interpolate.splrep(x, fx)

        f_rs_R500c = (500. * rho_c) / (200. * rho_m) * interpolate.splev(1./c200m, fx_intp)
        x_rs_R500c = interpolate.splev(f_rs_R500c, xf_intp)

        R500c = rs / x_rs_R500c
        M500c = (4. * np.pi / 3.) * np.power(R500c, 3.) * 500. * rho_c
        return  M500c


    def _get_integrated2D(self, pk_intp, **params_values_dict):

        zarr = self.zarr
        zz = self.zz
        marr = np.exp(self.marr)
        dlnm = self.dlnm
        Nq = self.Nq
        h = self.theory.get_param("H0") / 100.0

        dVdzdO = self._get_dVdzdO(zz)
        dndlnm = self._get_dndlnm(zz, pk_intp, **params_values_dict)
        surveydeg2 = self.skyfracs.sum()
        intgr = dndlnm * dVdzdO * surveydeg2
        intgr = intgr.T

        if self.theorypred['md_hmf'] != self.theorypred['md_ym']:
            if self.theorypred['choose_theory'] == 'CCL':
                mf_data = self.theory.get_nc_data()
                md_hmf = mf_data['md']

                if self.theorypred['md_ym'] == '200m':
                    md_ym = ccl.halos.MassDef200m(c_m='Bhattacharya13')
                elif self.theorypred['md_ym'] == '200c':
                    md_ym = ccl.halos.MassDef200c(c_m='Bhattacharya13')
                elif self.theorypred['md_ym'] == '500c':
                    md_ym = ccl.halos.MassDef(500, 'critical')
                else:
                    raise NotImplementedError('Only md_hmf = 200m, 200c and 500c currently supported.')

                cosmo = self.theory.get_CCL()['cosmo']
                a = 1./(1. + zz)
                marr_ymmd = np.array([md_hmf.translate_mass(cosmo, marr/h, ai, md_ym) for ai in a])*h
            else:
                if self.theorypred['md_hmf'] == '200m' and self.theorypred['md_ym'] == '500c':
                    marr_ymmd = self._get_M500c_from_M200m(marr, zz).T
                else:
                    raise NotImplementedError()
        else:
            marr_ymmd = marr

        if self.theorypred['md_ym'] != '500c':
            mf_data = self.theory.get_nc_data()
            md_hmf = mf_data['md']
            md_500c = ccl.halos.MassDef(500, 'critical')
            cosmo = self.theory.get_CCL()['cosmo']
            a = 1. / (1. + zz)
            marr_500c = np.array([md_hmf.translate_mass(cosmo, marr / h, ai, md_500c) for ai in a]) * h
        else:
            marr_500c = None

        y0 = self._get_y0(marr_ymmd, zz, marr_500c, **params_values_dict)
        # print('shape y0:',np.shape(y0))

        cc = []
        for kk in range(Nq):
            cc.append(self._get_completeness2D(marr, zz, y0, kk, **params_values_dict))
        cc = np.asarray(cc)
        # print('cc shape:',np.shape(cc))
        # for qq in range(np.shape(cc)[0]):
        #     print(qq,cc[qq][10])

        #nzarr = np.linspace(0, 2.8, 29)
        nzarr = np.linspace(0, 2.9, 30)

        delN2D = np.zeros((len(zarr), Nq))

        # print('zz:',zz)
        # print('zarr:',zarr)
        # print('nzarr:',nzarr)

        for kk in range(Nq):
            for i in range(len(zarr)):
                test = np.abs(zz - nzarr[i])
                i1 = np.argmin(test)
                test = np.abs(zz - nzarr[i+1])
                i2 = np.argmin(test)
                # if kk == 0:
                #     print('steps id min max :',i,i1, i2-1)
                zs = np.arange(i1, i2)

                sum = 0.
                sumzs = np.zeros(len(zz))
                for ii in zs:
                    for j in range(len(marr)):
                        sumzs[ii] += 0.5 * (intgr[ii,j]*cc[kk,ii,j] + intgr[ii+1,j]*cc[kk,ii+1,j]) * dlnm * (zz[ii+1] - zz[ii])
                        # sumzs[ii] += 0.5 * (intgr[ii,j] + intgr[ii+1,j]) * dlnm * (zz[ii+1] - zz[ii]) #NB no completness check

                    sum += sumzs[ii]

                delN2D[i,kk] = sum
        self.log.info("\r Total predicted 2D N = {}".format(delN2D.sum()))

        for i in range(len(zarr)):
            self.log.info('Number of clusters in redshift bin {}: {}.'.format(i, delN2D[i,:].sum()))
        self.log.info('------------')
        for kk in range(Nq):
            self.log.info('Number of clusters in snr bin {}: {}.'.format(kk, delN2D[:,kk].sum()))
        self.log.info("Total predicted 2D N = {}.".format(delN2D.sum()))

        return delN2D


    def _get_theory(self, pk_intp, **params_values_dict):

        start = t.time()

        if self.theorypred['choose_dim'] == '1D':
            delN = self._get_integrated(pk_intp, **params_values_dict)
        else:
            delN = self._get_integrated2D(pk_intp, **params_values_dict)

        elapsed = t.time() - start
        self.log.info("Theory N calculation took {} seconds.".format(elapsed))

        return delN


    # y-m scaling relation for completeness
    def _get_y0(self, mass, z, mass_500c, **params_values_dict):
        # print('mass_500c:',mass_500c)
        if mass_500c is None:
            mass_500c = mass

        A0 = params_values_dict["tenToA0"]
        B0 = params_values_dict["B0"]
        C0 = params_values_dict["C0"]
        bias = params_values_dict["bias_sz"]

        Ez = self._get_Ez(z)
        Ez = Ez[:,None]
        h = self.theory.get_param("H0") / 100.0

        mb = mass * bias
        mb_500c = mass_500c*bias
        #TODO: Is removing h correct here - matches Hasselfield but is different from before
        Mpivot = self.YM['Mpivot']*h  # convert to Msun/h.

        def theta(m):

            thetastar = 6.997
            alpha_theta = 1./3.
            DAz = self._get_DAz(z) * h
            DAz = DAz[:,None]
            H0 = self.theory.get_param("H0")
            ttstar = thetastar * (H0/70.)**(-2./3.)

            return ttstar*(m/MPIVOT_THETA/h)**alpha_theta * Ez**(-2./3.) * (100.*DAz/500/H0)**(-1.)

        def splQ(x):
            if self.selfunc['mode'] == 'single_tile' or self.selfunc['average_Q']:
                tck = interpolate.splrep(self.tt500, self.Q)
                newQ = interpolate.splev(x, tck)
            else:
                newQ = []
                for i in range(len(self.Q[0])):
                    tck = interpolate.splrep(self.tt500, self.Q[:,i])
                    newQ.append(interpolate.splev(x, tck))
            return np.asarray(np.abs(newQ))

        def rel(m):
            #mm = m / mpivot
            #t = -0.008488*(mm*Ez[:,None])**(-0.585)
            if self.theorypred['rel_correction']:
                t = -0.008488*(mm*Ez)**(-0.585) ###### M200m
                res = 1.+ 3.79*t - 28.2*(t**2.)
            else:
                res = 1.
            return res

        if self.selfunc['mode'] == 'single_tile' or self.selfunc['average_Q']:
            #y0 = A0 * (Ez[:,None]**2.) * (mb / mpivot)**(1. + B0) * splQ(theta(mb)) * rel(mb)
            y0 = A0 * (Ez**2.) * (mb / Mpivot)**(1. + B0) * splQ(theta(mb_500c)) #* rel(mb) ###### M200m
            y0 = y0.T ###### M200m
        else:
            y0 = A0 * (Ez ** 2.) * (mb / Mpivot) ** (1. + B0) * splQ(theta(mb_500c))
            # y0 = np.transpose(arg, axes=[1, 2, 0])

        # print('mb:',mb)
        # print('z:',z)
        # print('Ez:',Ez)
        if self.theorypred['use_class_sz']:
            ym_class_sz = []
            theta_class_sz = []
            for zz in z:
                ym_class_sz.append(self.get_y_at_m_and_z(mb,zz))
                theta_class_sz.append(self.get_theta_at_m_and_z(mb,zz))
            ym_class_sz = np.asarray([np.asarray(ym_class_sz)])
            # theta_class_sz = np.asarray([np.asarray(theta_class_sz)])
            # print('get_y_at_m_and_z shape:',np.shape(ym_class_sz))
            # print('get_theta_at_m_and_z shape:',np.shape(theta_class_sz))
            # print('y0 shape')


            # print('shape of theta(mb_500c):',np.shape(theta(mb_500c)))
            y0_class_sz = ym_class_sz*splQ(theta_class_sz)
            # print('shape y0_class_sz:',np.shape(y0_class_sz))
        # return y0
        if self.theorypred['use_class_sz']:
            return y0_class_sz
        else:
            return y0


    # completeness 2D
    def _get_completeness2D(self, marr, zarr, y0, qbin, **params_values_dict):

        scatter = params_values_dict["scatter_sz"]
        noise = self.noise
        qcut = self.qcut
        skyfracs = self.skyfracs/self.skyfracs.sum()
        Npatches = len(skyfracs)

        if self.selfunc['mode'] != 'single_tile' and not self.selfunc['average_Q']:
            if self.selfunc['mode'] == 'inpt_dwnsmpld':
                tile_list = self.tname
            elif self.selfunc['mode'] == 'downsample':
                tile_list = np.arange(noise.shape[0])+1
            elif self.selfunc['mode'] == 'full':
                tile_list = self.tile_list
        else:
            tile_list = None

        Nq = self.Nq
        qbins = self.qbins

        a_pool = multiprocessing.Pool()
        completeness = a_pool.map(partial(get_comp_zarr2D,
                                        Nm=len(marr),
                                        qcut=qcut,
                                        noise=noise,
                                        skyfracs=skyfracs,
                                        y0=y0,
                                        Nq=Nq,
                                        qbins=qbins,
                                        qbin=qbin,
                                        lnyy=None,
                                        dyy=None,
                                        yy=None,
                                        temp=None,
                                        mode=self.selfunc['mode'],
                                        compl_mode=self.theorypred['compl_mode'],
                                        tile=tile_list,
                                        average_Q=self.selfunc['average_Q'],
                                        scatter=scatter),range(len(zarr)))


        a_pool.close()
        comp = np.asarray(completeness)
        comp[comp < 0.] = 0.
        comp[comp > 1.] = 1.
        # comp[comp > 0.] = 1.

        return comp




def get_comp_zarr2D(index_z, Nm, qcut, noise, skyfracs, y0, Nq, qbins, qbin, lnyy, dyy, yy, temp, mode, compl_mode, average_Q, tile, scatter):

    kk = qbin
    qmin = qbins[kk]
    qmax = qbins[kk+1]

    res = []
    for i in range(Nm):
        erfunc = []
        for j in range(len(skyfracs)):
            erfunc.append(get_erf_compl(y0[int(tile[j])-1,index_z,i], qmin, qmax, noise[j], qcut))
        erfunc = np.asarray(erfunc)
        res.append(np.dot(erfunc, skyfracs))

    return res

def get_erf_compl(y, qmin, qmax, rms, qcut):

    arg1 = (y/rms - qmax)/np.sqrt(2.)
    if qmin > qcut:
        qlim = qmin
    else:
        qlim = qcut
    arg2 = (y/rms - qlim)/np.sqrt(2.)
    erf_compl = (special.erf(arg2) - special.erf(arg1)) / 2.
    return erf_compl

def get_erf(y, rms, cut):
    arg = (y - cut*rms)/np.sqrt(2.)/rms
    erfc = (special.erf(arg) + 1.)/2.
    return erfc

def roundup(x, places):
  d = np.power(10., places)
  if x < 0:
    return m.floor(x * d) / d
  else:
    return m.ceil(x * d) / d


def splintnr(xa, ya, y2a, n, xx):
    i = 0
    res = []
    for i in range(len(xx)):
        x = xx[i]
        klo = 1
        khi = n
        while khi - klo > 1 :
            k = int((khi + klo)/2.)
            if xa[k] >= x :
                khi = k
            else:
                klo = k
        else:
            h = xa[khi] - xa[klo]
            a = (xa[khi] - x)/h
            b = (x - xa[klo])/h
            y = a*ya[klo] + b*ya[khi] + ( (a**3. - a)*y2a[klo] + (b**3. - b)*y2a[khi]) * (h**2.)/6.
        res.append(y)
    return np.asarray(res)
