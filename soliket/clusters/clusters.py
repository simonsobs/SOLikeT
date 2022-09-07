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

import pyccl as ccl
from classy_sz import Class # TBD: change this import as optional

from ..poisson import PoissonLikelihood
from ..cash import CashCLikelihood
from . import massfunc as mf


import soliket.clusters.nemo_mocks as selfunc
from ..constants import MPC2CM, MSUN_CGS, G_CGS, C_M_S, T_CMB
from ..constants import h_Planck, k_Boltzmann, electron_mass_kg, elementary_charge


C_KM_S = 2.99792e5
MPIVOT_THETA = 3e14 # [Msun]
rho_crit0H100 = (3. / (8. * np.pi) * (100. * 1.e5) ** 2.) \
                    / G_CGS * MPC2CM / MSUN_CGS


def gaussian(xx, mu, sig, noNorm=False):
    if noNorm:
        return np.exp(-1.0 * (xx - mu) ** 2 / (2.0 * sig ** 2.0))
    else:
        return 1.0 / (sig * np.sqrt(2 * np.pi)) \
                            * np.exp(-1.0 * (xx - mu) ** 2 / (2.0 * sig ** 2.0))



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

        self.log.info('Initializing clusters.py (binned)')

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
        elif self.selfunc['mode'] == 'injection':
            self.log.info('Running injection based selection function.')

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
        debiasDOF = 0
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
        self.zbins = zbins

        self.log.info("Number of redshift bins = {}.".format(len(zarr)))

        # mass bin
        self.lnmmin = np.log(self.binning['M']['Mmin'])
        self.lnmmax = np.log(self.binning['M']['Mmax'])
        self.dlnm = self.binning['M']['dlogM']
        self.lnmarr = np.arange(self.lnmmin+(self.dlnm/2.), self.lnmmax, self.dlnm)
        # this is to be consist with szcounts.f90 - maybe switch to linspace?

        self.log.info('Number of mass bins for theory calculation {}.'.format(len(self.lnmarr)))
        #TODO: I removed the bin where everything is larger than zmax - is this ok?
        delNcat, _ = np.histogram(z, bins=zbins)

        self.delNcat = zarr, delNcat

        # SNR binning (following szcounts.f90)
        logqmin = self.binning['q']['log10qmin']
        logqmax = self.binning['q']['log10qmax']
        dlogq = self.binning['q']['dlog10q']

        # TODO: I removed the bin where everything is larger than qmax - is this ok?
        Nq = int((logqmax - logqmin)/dlogq) + 1

        # constant binning in log10
        qbins = np.arange(logqmin, logqmax+dlogq, dlogq)
        qarr = 10**(0.5*(qbins[:-1] + qbins[1:]))


        if dimension == "2D":
            self.log.info('The lowest SNR = {}.'.format(snr.min()))
            self.log.info('The highest SNR = {}.'.format(snr.max()))
            self.log.info('Number of SNR bins = {}.'.format(Nq))
            self.log.info('Edges of SNR bins = {}.'.format(qbins))

        delN2Dcat, _, _ = np.histogram2d(z, snr, bins=[zbins, 10**qbins])

        self.Nq = Nq
        self.qarr = qarr

        self.qbins = 10**qbins
        self.dlogq = dlogq
        self.delN2Dcat = zarr, qarr, delN2Dcat

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
            if self.selfunc['mode'] == 'injection':
                self.compThetaInterpolator = selfunc.get_completess_inj_theta_y(self.data_directory, self.qcut, self.qbins)
            elif self.selfunc['mode'] == 'inpt_dwnsmpld':
                self.log.info('Reading pre-downsampled Q function.')
                # for quick reading theta and Q data is saved first and just called
                self.datafile_Q = self.data['Q_file']
                Qfile = np.load(os.path.join(self.data_directory, self.datafile_Q))
                self.tt500 = Qfile['theta']
                self.Q = Qfile['Q']
                assert len(self.tt500) == len(self.Q[:,0])

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
                    self.Q = allQ
                else:
                    self.log.info('Reading in binned Q function from file.')
                    Qfile = np.load(datafile_Q_dwsmpld)
                    self.Q = Qfile['Q_dwsmpld']
                    self.tt500 = Qfile['tt500']

        self.log.info('Reading RMS.')
        if self.selfunc['mode'] == 'injection':
            self.log.info('Using completeness calculated using injection method.')
            self.datafile_rms = self.data['rms_file']
            list = fits.open(os.path.join(self.data_directory, self.datafile_rms))
            file_rms = list[1].data
            self.skyfracs = file_rms['areaDeg2'] * np.deg2rad(1.) ** 2

        elif self.selfunc['mode'] == 'single_tile':
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
                        filename_rms + 'dwsmpld_nbins={}'.format(self.selfunc['dwnsmpl_bins']) + '.npz')
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

                Qdwnsmpld = np.zeros((self.Q.shape[0], self.selfunc['dwnsmpl_bins']))

                for i in range(self.selfunc['dwnsmpl_bins']):
                    tempind = np.where(bin_ind == i + 1)[0]
                    if len(tempind) == 0:
                        self.log.info('Found empty bin.')
                        Qdwnsmpld[:, i] = np.zeros(self.Q.shape[0])
                    else:
                        temparea = self.skyfracs[tempind]
                        temptiles = self.tname[tempind]
                        test = [tiledict[key] for key in temptiles]
                        Qdwnsmpld[:, i] = np.average(self.Q[:, test], axis=1, weights=temparea)

                self.noise = 0.5*(binned_rms_edges[:-1] + binned_rms_edges[1:])
                self.skyfracs = binned_area
                self.Q = Qdwnsmpld
                self.log.info("Number of downsampled sky patches = {}.".format(self.skyfracs.size))

                assert self.noise.shape[0] == self.skyfracs.shape[0] and self.noise.shape[0] == self.Q.shape[1]

                if self.selfunc['save_dwsmpld']:
                    np.savez(datafile_Q_dwsmpld, Q_dwsmpld=Qdwnsmpld, tt500=self.tt500)
                    np.savez(datafile_rms_dwsmpld, noise=self.noise, skyfracs=self.skyfracs)

        elif self.selfunc['mode'] == 'full':
            tiledict = dict(zip(tilename, np.arange(tile_area[:, 0].shape[0])))
            self.tile_list = [tiledict[key]+1 for key in self.tname]

        if self.selfunc['mode'] != 'injection':
            if self.selfunc['average_Q']:
                self.Q = np.mean(self.Q, axis=1)
                self.log.info("Number of Q functions = {}.".format(self.Q.ndim))
                self.log.info("Using one averaged Q function for optimisation")
            else:
                self.Q = self.Q
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
        # print(" Nz for higher resolution = ", len(zz))


        super().initialize()

    def get_requirements(self):
        return get_requirements(self)

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
        marr = np.exp(self.lnmarr)
        Nq = self.Nq



        dVdzdO = get_dVdz(self,zz)

        h = self.theory.get_param("H0") / 100.0


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

        if self.selfunc['mode'] != 'injection':
            y0 = _get_y0(self,marr_ymmd, zz, marr_500c, **params_values_dict)
        else:
            y0 = None


        cc = []
        for kk in range(Nq):
            cc.append(self._get_completeness2D(marr, zz, y0, kk, marr_500c, **params_values_dict))
        cc = np.asarray(cc)

        delN2D = np.zeros((len(zarr), Nq))

        nzarr = self.zbins

        for kk in range(Nq):
            for i in range(len(zarr)):
                test = np.abs(zz - nzarr[i])
                i1 = np.argmin(test)
                test = np.abs(zz - nzarr[i+1])
                i2 = np.argmin(test)
                zs = np.arange(i1, i2)

                sum = 0.
                sumzs = np.zeros(len(zz))
                for ii in zs:
                    for j in range(len(marr)):
                        sumzs[ii] += 0.5 * (intgr[ii,j]*cc[kk,ii,j] + intgr[ii+1,j]*cc[kk,ii+1,j]) * self.dlnm * (zz[ii+1] - zz[ii])
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

    def get_completeness2D_inj(self, mass, z, mass_500c, qbin, **params_values_dict):

        y0 = _get_y0(self,mass, z, mass_500c, use_Q=False, **params_values_dict)
        theta = _theta(self,mass_500c, z)

        comp = np.zeros_like(theta)
        for i in range(theta.shape[0]):
            comp[i, :] = self.compThetaInterpolator[qbin](theta[i, :], y0[i, :]/1e-4, grid=False)
        comp[comp < 0] = 0
        return comp


    # completeness 2D
    def _get_completeness2D(self, marr, zarr, y0, qbin,  marr_500c=None, **params_values_dict):
        if self.selfunc['mode'] != 'injection':
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
        else:
            comp = self.get_completeness2D_inj(marr, zarr, marr_500c, qbin, **params_values_dict)
        return comp


class UnbinnedClusterLikelihood(PoissonLikelihood):
    name = "Unbinned Clusters"
    columns = ["tsz_signal", "z", "tsz_signal_err","tile_name"]

    verbose: bool = False
    data: dict = {}
    theorypred: dict = {}
    selfunc: dict = {}
    binning: dict = {}
    YM: dict = {}
    params = {"tenToA0":None, "B0":None, "C0":None, "scatter_sz":None, "bias_sz":None}

    def initialize(self):
        self.log = logging.getLogger('UnbinnedCluster')
        handler = logging.StreamHandler()
        self.log.addHandler(handler)
        self.log.propagate = False
        if self.verbose:
            self.log.setLevel(logging.INFO)
        else:
            self.log.setLevel(logging.ERROR)

        self.log.info('Initializing clusters.py (unbinned)')

        self.qcut = self.selfunc['SNRcut']

        self.LgY = np.arange(-6, -2.5, 0.01) # for integration over y when scatter != 0

        # reading catalogue
        self.log.info('Reading data catalog.')
        self.datafile = self.data['cat_file']
        self.data_directory = self.data['data_path']
        catf = fits.open(os.path.join(self.data_directory, self.datafile))
        data = catf[1].data
        zcat = data.field("redshift")
        qcat = data.field("fixed_SNR") #NB note that there are another SNR in the catalogue
        cat_tsz_signal = data.field("fixed_y_c")
        cat_tsz_signal_err = data.field("fixed_err_y_c")
        cat_tile_name = data.field("tileName")

        # to print all columns: print(catf[1].columns)
        catQ = data.field("Q")
        ind = np.where(qcat >= self.qcut)[0]

        self.z_cat = zcat[ind]
        self.cat_tsz_signal = cat_tsz_signal[ind]
        self.cat_tsz_signal_err = cat_tsz_signal_err[ind]
        self.cat_tile_name = cat_tile_name[ind]

        self.lnmmin = np.log(self.binning['M']['Mmin'])
        self.lnmmax = np.log(self.binning['M']['Mmax'])
        self.dlnm = self.binning['M']['dlogM']
        self.lnmarr = np.arange(self.lnmmin+(self.dlnm/2.), self.lnmmax, self.dlnm)

        self.zz = np.arange(0, 3, 0.05) # redshift bounds should correspond to catalogue
        self.k = np.logspace(-4, np.log10(5), 200)
        # self.mdef = ccl.halos.MassDef(500, 'critical')

        self.log.info('Using completeness calculated using injection method.')
        self.datafile_rms = self.data['rms_file']
        list = fits.open(os.path.join(self.data_directory, self.datafile_rms))
        file_rms = list[1].data
        self.skyfracs = file_rms['areaDeg2'] * np.deg2rad(1.) ** 2
        self.log.info('Entire survey area = {} deg2.'.format(self.skyfracs.sum()/(np.deg2rad(1.)**2.)))


        self.datafile_Q = self.data['Q_file']
        filename_Q, ext = os.path.splitext(self.datafile_Q)
        datafile_Q_dwsmpld = os.path.join(self.data_directory,
                             filename_Q + 'dwsmpld_nbins={}'.format(self.selfunc['dwnsmpl_bins']) + '.npz')
        if os.path.exists(datafile_Q_dwsmpld):
            self.log.info('Reading in binned Q function from file.')
            Qfile = np.load(datafile_Q_dwsmpld)
            self.Q = Qfile['Q_dwsmpld']
            self.tt500 = Qfile['tt500']
        # exit(0)

        else:
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
            self.Q = allQ

        self.log.info('Reading full RMS.')
        self.datafile_rms = self.datafile_rms
        filename_rms, ext = os.path.splitext(self.datafile_rms)
        datafile_rms_dwsmpld = os.path.join(self.data_directory,
                filename_rms + 'dwsmpld_nbins={}'.format(self.selfunc['dwnsmpl_bins']) + '.npz')
        datafile_tiles_dwsmpld = os.path.join(self.data_directory,
                'tile_names' + 'dwsmpld_nbins={}'.format(self.selfunc['dwnsmpl_bins']) + '.npy')

        if os.path.exists(datafile_rms_dwsmpld):
            rms = np.load(datafile_rms_dwsmpld)
            self.noise = rms['noise']
            self.skyfracs = rms['skyfracs']
            self.log.info("Number of rms bins = {}.".format(self.skyfracs.size))

            self.tiles_dwnsmpld = np.load(datafile_tiles_dwsmpld,allow_pickle='TRUE').item()

        else:
            self.log.info('Reading in full RMS table.')

            list = fits.open(os.path.join(self.data_directory, self.datafile_rms))
            file_rms = list[1].data

            self.noise = file_rms['y0RMS']
            self.skyfracs = self.skyfracs#file_rms['areaDeg2']*np.deg2rad(1.)**2
            self.tname = file_rms['tileName']
            self.log.info("Number of tiles = {}. ".format(len(np.unique(self.tname))))
            self.log.info("Number of sky patches = {}.".format(self.skyfracs.size))
            # exit(0)

            self.log.info('Downsampling RMS and Q function using {} bins.'.format(self.selfunc['dwnsmpl_bins']))
            binned_stat = scipy.stats.binned_statistic(self.noise, self.skyfracs, statistic='sum',
                                                       bins=self.selfunc['dwnsmpl_bins'])
            binned_area = binned_stat[0]
            binned_rms_edges = binned_stat[1]

            bin_ind = np.digitize(self.noise, binned_rms_edges)
            tiledict = dict(zip(tilename, np.arange(tile_area[:, 0].shape[0])))

            Qdwnsmpld = np.zeros((self.Q.shape[0], self.selfunc['dwnsmpl_bins']))
            tiles_dwnsmpld = {}

            for i in range(self.selfunc['dwnsmpl_bins']):
                tempind = np.where(bin_ind == i + 1)[0]
                if len(tempind) == 0:
                    self.log.info('Found empty bin.')
                    Qdwnsmpld[:, i] = np.zeros(self.Q.shape[0])
                else:
                    print('dowsampled rms bin ',i)
                    temparea = self.skyfracs[tempind]
                    print('areas of tiles in bin',temparea)
                    temptiles = self.tname[tempind]
                    print('names of tiles in bin',temptiles)
                    for t in temptiles:
                        tiles_dwnsmpld[t] = i

                    test = [tiledict[key] for key in temptiles]
                    Qdwnsmpld[:, i] = np.average(self.Q[:, test], axis=1, weights=temparea)

            self.noise = 0.5*(binned_rms_edges[:-1] + binned_rms_edges[1:])
            self.skyfracs = binned_area
            self.Q = Qdwnsmpld
            self.tiles_dwnsmpld = tiles_dwnsmpld
            print('len(tiles_dwnsmpld)',tiles_dwnsmpld)
            self.log.info("Number of downsampled sky patches = {}.".format(self.skyfracs.size))

            assert self.noise.shape[0] == self.skyfracs.shape[0] and self.noise.shape[0] == self.Q.shape[1]

            if self.selfunc['save_dwsmpld']:
                np.savez(datafile_Q_dwsmpld, Q_dwsmpld=Qdwnsmpld, tt500=self.tt500)
                np.savez(datafile_rms_dwsmpld, noise=self.noise, skyfracs=self.skyfracs)
                np.save(datafile_tiles_dwsmpld, self.tiles_dwnsmpld)

        self.qmin = self.qcut

        self.num_noise_bins = self.skyfracs.size


        self.frac_of_survey  = self.skyfracs
        self.fskytotal = self.skyfracs.sum()
        self.Ythresh = self.noise
        super().initialize()

    def get_requirements(self):
        return get_requirements(self)

    def _get_catalog(self):
        return get_catalog(self)


    def _get_rate_fn(self,pk_intp, **kwargs):

        z_arr = self.zz
        dndlnm = get_dndlnm(self,z_arr, pk_intp, **kwargs)

        param_vals = kwargs

        dn_dzdm_interp = scipy.interpolate.interp2d( self.zz, self.lnmarr, np.log(dndlnm), kind='linear',
        copy=True, bounds_error=False,
        fill_value=-np.inf)

        h = self.theory.get_param("H0") / 100.0



        def Prob_per_cluster(z,tsz_signal,tsz_signal_err,tile_name):
            self.log.info('computing prob per cluster for cluster: %.5e %.5e %.5e %s'%(z,tsz_signal,tsz_signal_err,tile_name))

            rms_bin_index = self.tiles_dwnsmpld[tile_name]
            Pfunc_ind = self.Pfunc_per(
                rms_bin_index,
                self.lnmarr,
                z,
                tsz_signal * 1e-4,
                tsz_signal_err * 1e-4,
                param_vals,
            )

            dn_dzdm = np.exp(dn_dzdm_interp(z,self.lnmarr))
            dn_dzdm = np.squeeze(dn_dzdm)


            ans = np.trapz(dn_dzdm * Pfunc_ind, dx=np.diff(self.lnmarr, axis=0), axis=0)
            return ans

        return Prob_per_cluster
        # Implement a function that returns a rate function (function of (tsz_signal, z))



    def _get_n_expected(self, pk_intp,**kwargs):
        dVdz = get_dVdz(self,self.zz)
        dndlnm = get_dndlnm(self,self.zz, pk_intp, **kwargs)

        Ntot = 0
        rms_index = 0
        for Yt, frac in zip(self.Ythresh, self.frac_of_survey):
            Pfunc = self.PfuncY(rms_index,Yt, self.lnmarr, self.zz, kwargs) # dim (m,z)
            N_z = np.trapz(
                dndlnm * Pfunc, dx=np.diff(self.lnmarr[:,None], axis=0), axis=0
            ) # dim (z)

            Np = (
                np.trapz(N_z * dVdz, x=self.zz)
                * 4.0
                * np.pi
                * self.fskytotal
                * frac
            )
            Ntot += Np
            rms_index += 1
        self.log.info("Number of clusters = %.5e"%Ntot)
        return Ntot

    def P_Yo(self, rms_bin_index,LgY, M, z, param_vals):

        Ma = np.outer(M, np.ones(len(LgY[0, :])))
        mass_500c = None
        y0_new = _get_y0(self,np.exp(Ma), z, mass_500c, use_Q=True, **param_vals)
        y0_new = y0_new[rms_bin_index]
        Ytilde = y0_new
        Y = 10 ** LgY

        numer = -1.0 * (np.log(Y / Ytilde)) ** 2
        ans = (
                1.0 / (param_vals["scatter_sz"] * np.sqrt(2 * np.pi)) *
                np.exp(numer / (2.0 * param_vals["scatter_sz"] ** 2))
        )
        return ans

    def P_Yo_vec(self, rms_index, LgY, M, z, param_vals):

        mass_500c = None
        y0_new = _get_y0(self,np.exp(M), z, mass_500c, use_Q=True, **param_vals)
        y0_new = y0_new[rms_index]
        Y = 10 ** LgY
        Ytilde = np.repeat(y0_new[:, :, np.newaxis], LgY.shape[2], axis=2)

        numer = -1.0 * (np.log(Y / Ytilde)) ** 2

        ans = (
                1.0 / (param_vals["scatter_sz"] * np.sqrt(2 * np.pi)) *
                np.exp(numer / (2.0 * param_vals["scatter_sz"] ** 2))
        )
        return ans

    def Y_erf(self, Y, Ynoise):
        qmin = self.qmin
        ans = Y * 0.0
        ans[Y - qmin * Ynoise > 0] = 1.0
        return ans

    def P_of_gt_SN(self, rms_index, LgY, MM, zz, Ynoise, param_vals):
        if param_vals['scatter_sz'] != 0:
            Y = 10 ** LgY

            Yerf = self.Y_erf(Y, Ynoise) # array of size dim Y
            sig_tr = np.outer(np.ones([MM.shape[0], # (dim mass)
                                        MM.shape[1]]), # (dim z)
                                        Yerf )

            sig_thresh = np.reshape(sig_tr,
                                    (MM.shape[0], MM.shape[1], len(Yerf)))

            LgYa = np.outer(np.ones([MM.shape[0], MM.shape[1]]), LgY)
            LgYa2 = np.reshape(LgYa, (MM.shape[0], MM.shape[1], len(LgY)))

            # replace nan with 0's:
            P_Y = np.nan_to_num(self.P_Yo_vec(rms_index,LgYa2, MM, zz, param_vals))
            ans = np.trapz(P_Y * sig_thresh, x=LgY, axis=2) * np.log(10) # why log10?

        else:
            mass_500c = None
            y0_new = _get_y0(self,np.exp(MM), zz, mass_500c, use_Q=True, **param_vals)
            y0_new = y0_new[rms_index]
            ans = y0_new * 0.0
            ans[y0_new - self.qmin *self.Ythresh[rms_index] > 0] = 1.0
            ans = np.nan_to_num(ans)

        return ans

    def PfuncY(self, rms_index, YNoise, M, z_arr, param_vals):
        LgY = self.LgY
        P_func = np.outer(M, np.zeros([len(z_arr)]))
        M_arr = np.outer(M, np.ones([len(z_arr)]))
        P_func = self.P_of_gt_SN(rms_index, LgY, M_arr, z_arr, YNoise, param_vals)
        return P_func

    def Y_prob(self, Y_c, LgY, YNoise):
        Y = 10 ** (LgY)

        ans = gaussian(Y, Y_c, YNoise)
        return ans

    def Pfunc_per(self, rms_bin_index,MM, zz, Y_c, Y_err, param_vals):
        if param_vals["scatter_sz"] != 0:
            LgY = self.LgY
            LgYa = np.outer(np.ones(len(MM)), LgY)
            P_Y_sig = self.Y_prob(Y_c, LgY, Y_err)
            P_Y = np.nan_to_num(self.P_Yo(rms_bin_index,LgYa, MM, zz, param_vals))
            ans = np.trapz(P_Y * P_Y_sig, LgY, np.diff(LgY), axis=1)
        else:
            mass_500c = None
            y0_new = _get_y0(self,np.exp(MM), zz, mass_500c, use_Q=True, **param_vals)
            y0_new = y0_new[rms_bin_index]
            LgY = np.log10(y0_new)
            P_Y_sig = np.nan_to_num(self.Y_prob(Y_c, LgY, Y_err))
            ans = P_Y_sig

        return ans

def get_dVdz(both,zarr):
    """dV/dzdOmega"""
    DA_z = both.theory.get_angular_diameter_distance(zarr)

    dV_dz = (
        DA_z**2
        * (1.0 + zarr) ** 2
        / (both.theory.get_Hubble(zarr) / C_KM_S)
    )
    h = both.theory.get_param("H0") / 100.0
    return dV_dz*h**3

def get_Ez(both,zarr):
    Ez_interp = interp1d(both.zz , both.theory.get_Hubble(both.zz) / both.theory.get_param("H0"))
    return Ez_interp(zarr)


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

    marr = self.lnmarr  # Mass in units of Msun/h

    if self.theorypred['massfunc_mode'] == 'internal':
        h = self.theory.get_param("H0")/100.0
        Ez = get_Ez(self,z)

        om = get_om(self)
        rhocrit0 = rho_crit0H100 # [h2 msun Mpc-3]

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



def get_catalog(both):


    df = pd.DataFrame(
        {
            "z": both.z_cat.byteswap().newbyteorder(),#both.survey.clst_z.byteswap().newbyteorder(),
            "tsz_signal": both.cat_tsz_signal.byteswap().newbyteorder(), #both.survey.clst_y0.byteswap().newbyteorder(),
            "tsz_signal_err": both.cat_tsz_signal_err.byteswap().newbyteorder(),#survey.clst_y0err.byteswap().newbyteorder(),
            "tile_name": both.cat_tile_name.byteswap().newbyteorder()#survey.clst_y0err.byteswap().newbyteorder(),

        }
    )

    return df



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




def _splQ(self, theta):
    if self.selfunc['mode'] == 'single_tile' or self.selfunc['average_Q']:
        tck = scipy.interpolate.splrep(self.tt500, self.Q)
        newQ = scipy.interpolate.splev(theta, tck)
    else:
        newQ = []
        for i in range(len(self.Q[0])):
            tck = scipy.interpolate.splrep(self.tt500, self.Q[:, i])
            newQ.append(scipy.interpolate.splev(theta, tck))
    return np.asarray(np.abs(newQ))

def _theta(self, mass_500c, z, Ez=None):

    thetastar = 6.997
    alpha_theta = 1. / 3.
    H0 = self.theory.get_param("H0")
    h = H0/100.0

    if Ez is None:
        Ez = get_Ez(self,z)
        Ez = Ez[:, None]

    # DAz = self.theory.get_angular_diameter_distance(z) * h #self._get_DAz(z) * h
    DAz_interp = interp1d(self.zz , self.theory.get_angular_diameter_distance(self.zz) * h)
    DAz = DAz_interp(z)
    try:
        DAz = DAz[:, None]
    except:
        DAz = DAz
    ttstar = thetastar * (H0 / 70.) ** (-2. / 3.)

    if self.name == "Unbinned Clusters":
        Ez = Ez.T
        DAz = DAz.T

    return ttstar * (mass_500c / MPIVOT_THETA / h) ** alpha_theta * Ez ** (-2. / 3.) * (100. * DAz / 500 / H0) ** (-1.)


# y-m scaling relation for completeness
def _get_y0(self, mass, z, mass_500c, use_Q=True, **params_values_dict):
    if mass_500c is None:
        mass_500c = mass

    A0 = params_values_dict["tenToA0"]
    B0 = params_values_dict["B0"]
    C0 = params_values_dict["C0"]
    bias = params_values_dict["bias_sz"]

    Ez = get_Ez(self,z)
    try:
        Ez = Ez[:,None]
    except:
        Ez = Ez

    h = self.theory.get_param("H0") / 100.0

    mb = mass* bias
    mb_500c = mass_500c*bias

    Mpivot = self.YM['Mpivot']*h  # convert to Msun/h.

    def rel(m):
        #mm = m / mpivot
        #t = -0.008488*(mm*Ez[:,None])**(-0.585)
        if self.theorypred['rel_correction']:
            t = -0.008488*(mm*Ez)**(-0.585) ###### M200m
            res = 1.+ 3.79*t - 28.2*(t**2.)
        else:
            res = 1.
        return res

    if use_Q is True:
        theta = _theta(self,mb_500c, z, Ez)
        splQ = _splQ(self,theta)
    else:
        splQ = 1.


    if self.selfunc['mode'] == 'single_tile' or self.selfunc['average_Q']:
        y0 = A0 * (Ez**2.) * (mb / Mpivot)**(1. + B0) * splQ
        y0 = y0.T ###### M200m
    else:
        if self.name == "Unbinned Clusters":
            Ez = Ez.T

        y0 = A0 * (Ez ** 2.) * (mb / Mpivot) ** (1. + B0) * splQ
        # print('shape y0',np.shape(y0))

    return y0
