"""
requires extra: astlib,fits,os,sys,nemo
"""
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pkg_resources import resource_filename
import logging
from astropy.io import fits
import os, sys
import nemo as nm # needed for reading Q-functions
import scipy.stats # needed for binning rms
import scipy.interpolate
import scipy.integrate
import scipy.special
import time # for timing
import multiprocessing
from functools import partial

# import pyccl as ccl

from ..poisson import PoissonLikelihood
from ..cash import CashCLikelihood
from . import massfunc as mf
from .survey import SurveyData
from .sz_utils import szutils

C_KM_S = 2.99792e5


class SZModel:
    pass


class BinnedClusterLikelihood(CashCLikelihood):
    name = "Binned Clusters"

    data: dict = {}
    theorypred: dict = {}
    YM: dict = {}
    selfunc: dict = {}
    binning: dict = {}
    verbose: bool = False

    params = {"tenToA0":None, "B0":None, "C0":None, "scatter_sz":None, "bias_sz":None}


    def initialize(self):

        # self.zarr = np.arange(0, 2, 0.05)

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

    def _get_data(self):
        return self.delN2Dcat



    def _get_theory(self, pk_intp, **params_values_dict):
        start = time.time()

        delN = self._get_integrated2D(pk_intp, **params_values_dict)

        elapsed = time.time() - start
        self.log.info("Theory N calculation took {} seconds.".format(elapsed))

        return delN



    def _get_integrated2D(self, pk_intp, **params_values_dict):

        zarr = self.zarr
        zz = self.zz
        marr = np.exp(self.marr)
        dlnm = self.dlnm
        Nq = self.Nq
        h = self.theory.get_param("H0") / 100.0


        dVdzdO = get_dVdz(self,zz)*h**3

        # h = self.theory.get_param("H0") / 100.0
        # dVdzdO = (c_ms/1e3)*(((1. + self.zarr)*dAz)**2.)/Hz
        # return dVdzdO * h**3.

        dndlnm = get_dndlnm(self,zz, pk_intp, **params_values_dict)


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
        print('y0 needed:',y0)
        y0_nick = 0
        print('y0 nick: sort this out!',y0_nick)
        # print('shape y0:',np.shape(y0))
        # exit(0)

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

    # y-m scaling relation for completeness
    # needs to be syncronized with unbinned !
    def _get_y0(self, mass, z, mass_500c, **params_values_dict):
        # print('mass_500c:',mass_500c)
        if mass_500c is None:
            mass_500c = mass

        A0 = params_values_dict["tenToA0"]
        B0 = params_values_dict["B0"]
        C0 = params_values_dict["C0"]
        bias = params_values_dict["bias_sz"]

        Ez = get_Ez(self,z)
        Ez = Ez[:,None]
        h = self.theory.get_param("H0") / 100.0

        mb = mass * bias
        mb_500c = mass_500c*bias
        #TODO: Is removing h correct here - matches Hasselfield but is different from before
        Mpivot = self.YM['Mpivot']*h  # convert to Msun/h.

        def theta(m):

            thetastar = 6.997
            alpha_theta = 1./3.
            DAz = self.theory.get_angular_diameter_distance(z) * h
            DAz = DAz[:,None]
            H0 = self.theory.get_param("H0")
            ttstar = thetastar * (H0/70.)**(-2./3.)

            return ttstar*(m/szutils.MPIVOT_THETA/h)**alpha_theta * Ez**(-2./3.) * (100.*DAz/500/H0)**(-1.)

        def splQ(x):
            if self.selfunc['mode'] == 'single_tile' or self.selfunc['average_Q']:
                tck = scipy.interpolate.splrep(self.tt500, self.Q)
                newQ = scipy.interpolate.splev(x, tck)
            else:
                newQ = []
                for i in range(len(self.Q[0])):
                    tck = scipy.interpolate.splrep(self.tt500, self.Q[:,i])
                    newQ.append(scipy.interpolate.splev(x, tck))
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








class UnbinnedClusterLikelihood(PoissonLikelihood):
    name = "Unbinned Clusters"
    columns = ["tsz_signal", "z", "tsz_signal_err"]
    data_path = resource_filename("soliket", "clusters/data/selFn_equD56")
    # data_path = resource_filename("soliket", "clusters/data/selFn_SO")
    data_name = resource_filename("soliket", "clusters/data/E-D56Clusters.fits")
    # data_name = resource_filename("soliket",
    #                   "clusters/data/MFMF_WebSkyHalos_A10tSZ_3freq_tiles_mass.fits")
    theorypred: dict = {}

    def initialize(self):
        self.zarr = np.arange(0, 2, 0.05)
        self.k = np.logspace(-4, np.log10(5), 200)
        # self.mdef = ccl.halos.MassDef(500, 'critical')

        super().initialize()

    def get_requirements(self):
        return {
            "Pk_interpolator": {
                "z": self.zarr,
                "k_max": 5.0,
                "nonlinear": False,
                "hubble_units": False,  # cobaya told me to
                "k_hunit": False,  # cobaya told me to
                "vars_pairs": [["delta_nonu", "delta_nonu"]],
            },
            "Hubble": {"z": self.zarr},
            "angular_diameter_distance": {"z": self.zarr},
            "comoving_radial_distance": {"z": self.zarr}
            # "CCL": {"methods": {"sz_model": self._get_sz_model}, "kmax": 10},
        }

    def _get_sz_model(self, cosmo):
        model = SZModel()
        model.hmf = ccl.halos.MassFuncTinker08(cosmo, mass_def=self.mdef)
        model.hmb = ccl.halos.HaloBiasTinker10(
            cosmo, mass_def=self.mdef, mass_def_strict=False
        )
        model.hmc = ccl.halos.HMCalculator(cosmo, model.hmf, model.hmb, self.mdef)
        model.szk = SZTracer(cosmo)
        return model

    def _get_catalog(self):
        self.survey = SurveyData(
            self.data_path, self.data_name
        )  # , MattMock=False,tiles=False)

        self.szutils = szutils(self.survey)

        df = pd.DataFrame(
            {
                "z": self.survey.clst_z.byteswap().newbyteorder(),
                "tsz_signal": self.survey.clst_y0.byteswap().newbyteorder(),
                "tsz_signal_err": self.survey.clst_y0err.byteswap().newbyteorder(),
            }
        )
        return df

    # def _get_om(self):
    #     return (self.theory.get_param("omch2") + self.theory.get_param("ombh2")) / (
    #         (self.theory.get_param("H0") / 100.0) ** 2
    #     )

    def _get_ob(self):
        return (self.theory.get_param("ombh2")) / (
            (self.theory.get_param("H0") / 100.0) ** 2
        )

    # def _get_Ez(self):
    #     return self.theory.get_Hubble(self.zarr) / self.theory.get_param("H0")

    # NOT GOOD!
    def _get_Ez_interpolator(self):
        return interp1d(self.zarr, get_Ez(self,self.zarr))

    def _get_DAz(self):
        return self.theory.get_angular_diameter_distance(self.zarr)

    def _get_DAz_interpolator(self):
        return interp1d(self.zarr, self._get_DAz())

    def _get_HMF(self):
        h = self.theory.get_param("H0") / 100.0

        Pk_interpolator = self.theory.get_Pk_interpolator(
            ("delta_nonu", "delta_nonu"), nonlinear=False
        ).P
        pks = Pk_interpolator(self.zarr, self.k)
        # pkstest = Pk_interpolator(0.125, self.k )
        # print (pkstest * h**3 )

        Ez = (
            get_Ez(self,self.zarr)
        )  # self.theory.get_Hubble(self.zarr) / self.theory.get_param("H0")
        om = get_om(self)

        hmf = mf.HMF(om, Ez, pk=pks * h**3, kh=self.k / h, zarr=self.zarr)

        return hmf

    def _get_param_vals(self, **kwargs):
        # Read in scaling relation parameters
        # scat = kwargs['scat']
        # massbias = kwargs['massbias']
        # B0 = kwargs['B']
        B0 = 0.08
        scat = 0.2
        massbias = 1.0

        H0 = self.theory.get_param("H0")
        ob = self._get_ob()
        om = get_om(self)
        param_vals = {
            "om": om,
            "ob": ob,
            "H0": H0,
            "B0": B0,
            "scat": scat,
            "massbias": massbias,
        }
        return param_vals

    def _get_rate_fn(self, **kwargs):
        HMF = self._get_HMF()
        param_vals = self._get_param_vals(**kwargs)

        Ez_fn = self._get_Ez_interpolator()
        DA_fn = self._get_DAz_interpolator()

        dn_dzdm_interp = HMF.inter_dndmLogm(delta=500)

        h = self.theory.get_param("H0") / 100.0

        def Prob_per_cluster(z, tsz_signal, tsz_signal_err):
            c_y = tsz_signal
            c_yerr = tsz_signal_err
            c_z = z

            Pfunc_ind = self.szutils.Pfunc_per(
                HMF.M, c_z, c_y * 1e-4, c_yerr * 1e-4, param_vals, Ez_fn, DA_fn
            )

            dn_dzdm = 10 ** np.squeeze(dn_dzdm_interp(c_z, np.log10(HMF.M))) * h**4.0

            ans = np.trapz(dn_dzdm * Pfunc_ind, dx=np.diff(HMF.M, axis=0), axis=0)
            return ans

        return Prob_per_cluster
        # Implement a function that returns a rate function (function of (tsz_signal, z))



    def _get_n_expected(self, **kwargs):
        # def Ntot_survey(self,int_HMF,fsky,Ythresh,param_vals):

        HMF = self._get_HMF()
        param_vals = self._get_param_vals(**kwargs)
        Ez_fn = self._get_Ez_interpolator()
        DA_fn = self._get_DAz_interpolator()

        z_arr = self.zarr

        h = self.theory.get_param("H0") / 100.0

        Ntot = 0

        dVdz = get_dVdz(self,z_arr)

        dn_dzdm = HMF.dn_dM(HMF.M, 500.0) * h**4.0  # getting rid of hs

        for Yt, frac in zip(self.survey.Ythresh, self.survey.frac_of_survey):
            Pfunc = self.szutils.PfuncY(Yt, HMF.M, z_arr, param_vals, Ez_fn, DA_fn)
            N_z = np.trapz(
                dn_dzdm * Pfunc, dx=np.diff(HMF.M[:, None] / h, axis=0), axis=0
            )
            Ntot += (
                np.trapz(N_z * dVdz, x=z_arr)
                * 4.0
                * np.pi
                * self.survey.fskytotal
                * frac
            )

        return Ntot

    def _test_n_tot(self, **kwargs):

        HMF = self._get_HMF()
        param_vals = self._get_param_vals(**kwargs)
        Ez_fn = self._get_Ez_interpolator()
        DA_fn = self._get_DAz_interpolator()

        z_arr = self.zarr

        h = self.theory.get_param("H0") / 100.0

        Ntot = 0
        dVdz = get_dVdz(self,z_arr)
        dn_dzdm = HMF.dn_dM(HMF.M, 500.0) * h**4.0  # getting rid of hs
        # Test Mass function against Nemo.
        Pfunc = 1.0
        N_z = np.trapz(dn_dzdm * Pfunc, dx=np.diff(HMF.M[:, None] / h, axis=0), axis=0)
        Ntot = (
            np.trapz(N_z * dVdz, x=z_arr)
            * 4.0
            * np.pi
            * (600.0 / (4 * np.pi * (180 / np.pi) ** 2))
        )

        return Ntot



def get_dVdz(both,zarr):
    """dV/dzdOmega"""
    DA_z = both.theory.get_angular_diameter_distance(zarr)

    dV_dz = (
        DA_z**2
        * (1.0 + zarr) ** 2
        / (both.theory.get_Hubble(zarr) / C_KM_S)
    )

    # dV_dz *= (self.theory.get_param("H0") / 100.0) ** 3.0  # was h0
    return dV_dz

def get_Ez(both,zarr):
    return both.theory.get_Hubble(zarr) / both.theory.get_param("H0")


def get_om(both):
    if both.theorypred['choose_theory'] == "camb":
        om = (both.theory.get_param("omch2") + both.theory.get_param("ombh2") +
              both.theory.get_param("omnuh2"))/((both.theory.get_param("H0")/100.0)**2)
    elif both.theorypred['choose_theory'] == "class":
        om = (both.theory.get_param("omega_cdm") +
              both.theory.get_param("omega_b"))/((both.theory.get_param("H0")/100.0)**2) # for CLASS
    else:
        print('please specify theory: camb/class')
        exit(0)
    return om




def get_dndlnm(self, z, pk_intp, **params_values_dict):

    #TODO: Why is zarr not used?
    # zarr = self.zarr
    marr = self.marr  # Mass in units of Msun/h

    if self.theorypred['massfunc_mode'] == 'internal':
        h = self.theory.get_param("H0")/100.0
        Ez = get_Ez(self,z)

        om = get_om(self)
        rhocrit0 = szutils.rho_crit0H100 # [h2 msun Mpc-3]

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
                tck = scipy.interpolate.splrep(zpk, pks0[:,i])
                newp[:,i] = scipy.interpolate.splev(newz, tck)
            return newp

        # rebin
        pks = pks_zbins(z)

        pks *= h**3.
        kh = k/h

        def radius(M): # R in units of Mpc/h
            return (0.75*M/np.pi/rhom0)**(1./3.)

        def win(x):
            return 3.*(np.sin(x) - x*np.cos(x))/(x**3.)

        def win_prime(x):
            return 3.*np.sin(x)/(x**2.) - 9.*(np.sin(x) - x*np.cos(x))/(x**4.)

        def sigma_sq(R, k):
            integral = np.zeros((len(k), len(marr), len(z)))
            for i in range(k.size):
                integral[i,:,:] = np.array((k[i]**2.)*pks[:,i]*(win(k[i]*R)**2.))
            return scipy.integrate.simps(integral, k, axis=0)/(2.*np.pi**2.)

        def sigma_sq_prime(R, k):
            # this is derivative of sigmaR squared
            # so 2 * sigmaR * dsigmaR/dR
            integral = np.zeros((len(k), len(marr), len(z)))
            for i in range(k.size):
                integral[i,:,:] = np.array((k[i]**2.)*pks[:,i]*2.*k[i]*win(k[i]*R)*win_prime(k[i]*R))
            return scipy.integrate.simps(integral, k, axis=0)/(2.*np.pi**2.)

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

            tck1 = scipy.interpolate.splrep(delta, par_aa)
            tck2 = scipy.interpolate.splrep(delta, par_a)
            tck3 = scipy.interpolate.splrep(delta, par_b)
            tck4 = scipy.interpolate.splrep(delta, par_c)

            par1 = scipy.interpolate.splev(np.log10(dsoz), tck1)
            par2 = scipy.interpolate.splev(np.log10(dsoz), tck2)
            par3 = scipy.interpolate.splev(np.log10(dsoz), tck3)
            par4 = scipy.interpolate.splev(np.log10(dsoz), tck4)

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
        return hmf_internal

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


### check these in szutils in some form??
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
    erf_compl = (scipy.special.erf(arg2) - scipy.special.erf(arg1)) / 2.
    return erf_compl
