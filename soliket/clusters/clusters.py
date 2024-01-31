"""
requires extra: astlib,fits,os,sys,nemo
=======
.. module:: clusters

:Synopsis: Poisson likelihood for SZ clusters for Simons Osbervatory 
:Authors: Nick Battaglia, Eunseong Lee

Likelihood for unbinned tSZ galaxy cluster number counts. Currently under development and
should be used only with caution and advice. Uses the SZ scaling relations from 
Hasselfield et al (2013) [1]_ to compare observed number of :math:`y`-map detections 
with the prediction from a Tinker [2]_ Halo Mass Function.

References
----------
.. [1] Hasselfield et al, JCAP 07, 008 (2013) `arXiv:1301.0816 
                                                <https://arxiv.org/abs/1301.0816>`_
.. [2] Tinker et al, Astrophys. J. 688, 2, 709 (2008) `arXiv:0803.2706 
                                                    <https://arxiv.org/abs/0803.2706>`_

"""

import numpy as np
import pandas as pd
import nemo as nm # needed for reading Q-functions
import logging
import os, sys
import time # for timing
from scipy import special, stats, interpolate, integrate
from scipy.interpolate import interp1d, interp2d
from astropy.io import fits
import astropy.table as atpy
import pyccl as ccl
import soliket.clusters.nemo_mocks as selfunc

from ..poisson import PoissonLikelihood
from ..cash import CashCLikelihood
from ..constants import MPC2CM, MSUN_CGS, G_CGS
#from classy_sz import Class # TBD: change this import as optional

C_KM_S = 2.99792e5
MPIVOT_THETA = 3e14 # [Msun]
rho_crit0H100 = (3. / (8. * np.pi) * (100. * 1.e5) ** 2.) / G_CGS * MPC2CM / MSUN_CGS


class BinnedClusterLikelihood(CashCLikelihood):
    name = "Binned Clusters"

    data: dict = {}
    theorypred: dict = {}
    YM: dict = {}
    selfunc: dict = {}
    binning: dict = {}
    verbose: bool = False

    params = {"tenToA0":None, "B0":None, "C0":None, "scatter_sz":None, "bias_sz":None,
              "opt_bias_efold":None, "opt_bias_ped":None, "opt_bias_norm":None}

    def initialize(self):

        # constant binning in log10
        # WARNING: This seems brittle
        # qbins = np.arange(self.binning['q']['log10qmin'], self.binning['q']['log10qmax']+self.binning['q']['dlog10q'], self.binning['q']['dlog10q'])
        # self.qbins = 10**qbins
        # self.qarr = 10**(0.5*(qbins[:-1] + qbins[1:]))
        # self.Nq = int((self.binning['q']['log10qmax'] - self.binning['q']['log10qmin'])/self.binning['q']['dlog10q']) + 1
        # Revised binning
        self.Nq = int(np.ceil((self.binning['q']['log10qmax']-self.binning['q']['log10qmin'])/self.binning['q']['dlog10q']))
        qbins=np.linspace(self.binning['q']['log10qmin'], self.binning['q']['log10qmax'], self.Nq)
        self.qbins = np.power(10, qbins) # bin edges
        self.qarr = np.power(10, (qbins[1:]+qbins[:-1])/2)
        self.Nq = self.Nq-1
        assert(self.Nq > 0)

        # this is for liklihood computation
        self.zcut = self.binning['exclude_zbin']

        initialize_common(self)

        delNcat, _ = np.histogram(self.z_cat, bins=self.zbins)
        self.delNcat = self.zarr, delNcat

        self.log.info("Number of redshift bins = {}.".format(len(self.zarr)))
        self.log.info('Number of SNR bins = {}.'.format(self.Nq))

        delN2Dcat, _, _ = np.histogram2d(self.z_cat, self.q_cat, bins=[self.zbins, self.qbins])
        self.delN2Dcat = self.zarr, self.qarr, delN2Dcat, self.zcut

        # finner binning for low redshift
        minz = self.zbins[0]
        maxz = self.zbins[-1]
        if minz < 0: minz = 0.0
        zi = minz

        # counting redshift bins
        Nzz = 0
        while zi <= maxz :
            zi = self._get_hres_z(zi)
            Nzz += 1

        zi = minz
        zz = np.zeros(Nzz)
        for i in range(Nzz):
            zz[i] = zi
            zi = self._get_hres_z(zi)
        if zz[0] == 0. : zz[0] = 1e-5
        self.zz = zz

        self.log.info('Number of redshift points for theory calculation = {}.'.format(len(self.zz)))
        self.log.info('Number of mass points for theory calculation = {}.'.format(len(self.lnmarr)))
        self.log.info('Number of y0 points for theory calculation = {}.'.format(len(self.lny)))

        super().initialize()

    def get_requirements(self):
        return get_requirements(self)

    def _get_hres_z(self, zi):
        # bins in redshifts are defined with higher resolution for low redshift
        # for now using the same binning as in NEMO for comparison
        hr = 0.2
        if zi < hr :
            dzi = 1e-2 #1e-3
        elif zi >= hr and zi <=1.:
            dzi = 1e-2
        else:
            dzi = 1e-2 #5e-2 #self.binning['z']['dz']
        hres_z = zi + dzi
        return hres_z

    def _get_data(self):
        return self.delN2Dcat

    def _get_theory(self, pk_intp, **kwargs):

        start = time.time()
        delN = self._get_integrated(pk_intp, **kwargs)
        elapsed = time.time() - start
        self.log.info("Theory N calculation took {:.3f} seconds.".format(elapsed))

        return delN

    def _get_integrated(self, pk_intp, **kwargs):

        zarr = self.zarr
        zz = self.zz
        marr = np.exp(self.lnmarr)
        Nq = self.Nq

        h = self.theory.get_param("H0") / 100.0

        dndlnm = get_dndlnm(self, zz, pk_intp)
        dVdzdO = get_dVdz(self, zz, dVdz_interp=False)
        surveydeg2 = self.skyfracs.sum()
        intgr = dndlnm * dVdzdO * surveydeg2
        intgr = intgr.T

        if self.theorypred['md_hmf'] != self.theorypred['md_ym']:
            marr_ymmd = convert_masses(self, marr, zz)
        else:
            marr_ymmd = marr

        if self.theorypred['md_ym'] != '500c':
            marr_500c = get_m500c(self, marr, zz)
        else:
            marr_500c = marr_ymmd

        if self.selfunc['method'] == 'SNRbased':
            y0 = get_y0(self, marr_ymmd, zz, marr_500c, use_Q=True, Ez_interp=False, **kwargs)
        else:
            y0 = None

        cc = np.array([self._get_completeness(marr, zz, y0, kk, marr_500c, **kwargs) for kk in range(Nq)])

        nzarr = self.zbins
        delN2D = np.zeros((len(zarr), Nq))

        # integrate over mass
        dndzz = np.trapz(intgr[None, :,:]*cc, dx=self.dlnm, axis=2)
        #dndzz = np.trapz(intgr[None, :,:]*cc, x=self.lnmarr, axis=2)

        # integrate over fine z bins and sum over in larger z bins
        for i in range(len(zarr)):

            test = np.abs(zz - nzarr[i])
            i1 = np.argmin(test)
            test = np.abs(zz - nzarr[i+1])
            i2 = np.argmin(test)

            delN2D[i,:] = np.trapz(dndzz[:, i1:i2+1], x=zz[i1:i2+1]).T

        self.log.info("\r Total predicted 2D N = {}".format(delN2D.sum()))

        for i in range(len(zarr)):
            self.log.info('Number of clusters in redshift bin {}: {}.'.format(i, delN2D[i,:].sum()))
        self.log.info('------------')
        for kk in range(Nq):
            self.log.info('Number of clusters in snr bin {}: {}.'.format(kk, delN2D[:,kk].sum()))
        self.log.info("Total predicted 2D N = {}.".format(delN2D.sum()))

        # Debugging
        # from astLib import astImages
        # astImages.saveFITS("theory_%.2f_%.2f.fits" % (self.theory.get_param('sigma8'), self.theory.get_param('Omega_c')), delN2D.transpose())
        # astImages.saveFITS("obs_%.2f_%.2f.fits" % (self.theory.get_param('sigma8'), self.theory.get_param('Omega_c')), self.delN2Dcat[2].transpose())
        # astImages.saveFITS("diff_%.2f_%.2f.fits" % (self.theory.get_param('sigma8'), self.theory.get_param('Omega_c')), (self.delN2Dcat[2]-delN2D).transpose())
        # print("inspect grid")
        # import IPython
        # IPython.embed()
        # sys.exit()

        return delN2D

    def _get_completeness_inj(self, mass, z, mass_500c, qbin, **params):

        scatter = params["scatter_sz"]

        y0 = get_y0(self, mass, z, mass_500c, use_Q=False, Ez_interp=False, **params)
        theta = get_theta(self, mass_500c, z)

        if scatter == 0:
            comp = np.zeros_like(theta)
            for i in range(theta.shape[0]):
                comp[i, :] = self.compThetaInterpolator[qbin](theta[i, :], y0[i, :]/1e-4, grid=False)
            comp[comp < 0] = 0

        else:
            comp = np.zeros((theta.shape[0], theta.shape[1], self.lny.shape[0]))
            for i in range(theta.shape[0]):
                comp[i, :] = self.compThetaInterpolator[qbin](theta[i, :], np.exp(self.lny)/1e-4, grid=True)
            comp[comp < 0] = 0

            fac = 1. / np.sqrt(2. * np.pi * scatter ** 2)
            arg = (self.lny[None, None, :] - np.log(y0[:, :, None])) / (np.sqrt(2.) * scatter)
            PY = fac * np.exp(-arg ** 2.)
            comp = np.trapz(comp*PY, self.lny, axis=-1)

        return comp


    def _get_completeness(self, marr, zarr, y0, qbin, marr_500c=None, **params):
        """Calculate completeness on (mass, z) grid for given signal-to-noise-bin.

        Args:
            marr (array): Masses defining the (mass, z) grid
            zarr (array): Redshifts defining the (mass, z) grid
            y0 (array): Cube of predicted y0 values on the (mass, z) grid - each
                plane corresponds to a different patch.
            qbin (int): Index of the signal-to-noise (q) bin for which completeness
                will be calculated.

        Returns:
            Completeness on (mass, z) grid for given signal to noise bin, averaged
            over all patches.

        """

        if self.selfunc['method'] == 'SNRbased':

            scatter = params["scatter_sz"]
            noise = self.noise
            qcut = self.qcut
            skyfracs = self.skyfracs/self.skyfracs.sum()
            Npatches = len(skyfracs)
            compl_mode = self.theorypred['compl_mode']

            Nq = self.Nq
            qbins = self.qbins
            kk = qbin
            qmin = qbins[kk]
            qmax = qbins[kk+1]
            opt_bias_corr_factor=np.ones(y0.shape)
            if self.selfunc['bias_handler'] == 'theory':
                for i in range(Npatches):
                    trueSNR=y0[i]/noise[i]
                    # bias_pars=self.selfunc['bias_model_params']
                    bias_pars={'efold': params['opt_bias_efold'], 'ped': params['opt_bias_ped'], 'norm': params['opt_bias_norm']}
                    opt_bias_corr_factor[i]=_opt_bias_func(trueSNR, bias_pars['efold'], bias_pars['ped'], bias_pars['norm'])

            if scatter == 0.:

                arg = []
                for i in range(Npatches):

                    if compl_mode == 'erf_prod':
                        arg.append(get_erf_prod(y0[i]*opt_bias_corr_factor[i], noise[i], qmin, qmax, qcut, kk, Nq))
                        #arg.append(get_stf_prod(y0[i], noise[i], qmin, qmax, qcut, kk, Nq))
                    elif compl_mode == 'erf_diff':
                        arg.append(get_erf_diff(y0[i]*opt_bias_corr_factor[i], noise[i], qmin, qmax, qcut))
                        #arg.append(get_stf_diff(y0[i], noise[i], qmin, qmax, qcut))

                comp = np.einsum('ijk,i->jk', np.nan_to_num(arg), skyfracs)

            else:

                lnyy = np.float32(self.lny)
                yy0 = np.exp(lnyy)
                mu = np.float32(np.log(y0*opt_bias_corr_factor))
                fac = np.float32(1./np.sqrt(2.*np.pi*scatter**2))
                comp = 0.
                for i in range(Npatches):

                    if compl_mode == 'erf_prod':
                        arg = get_erf_prod(yy0, noise[i], qmin, qmax, qcut, kk, Nq)
                        #arg = get_stf_prod(yy0, noise[i], qmin, qmax, qcut, kk, Nq)
                    elif compl_mode == 'erf_diff':
                        arg = get_erf_diff(yy0, noise[i], qmin, qmax, qcut)
                        #arg = get_stf_diff(yy0, noise[i], qmin, qmax, qcut)

                    cc = np.float32(arg * skyfracs[i])
                    arg0 = np.float32((lnyy[:, None,None] - mu[i])/(np.sqrt(2.)*scatter))
                    args = fac * np.exp(np.float32(-arg0**2.)) * cc[:, None,None]
                    comp += np.trapz(np.float32(args), x=lnyy, axis=0)

            comp[comp < 0.] = 0.
            comp[comp > 1.] = 1.

        else:
            comp = self._get_completeness_inj(marr, zarr, marr_500c, qbin, **params)
        return comp


def _opt_bias_func(snr, snrFold, pedestal, norm):
    """Return optimization bias correction factor - multiply true y0 by this to get what the cluster finder recovers

    """
    return norm*np.exp(-snr/snrFold)+pedestal


class UnbinnedClusterLikelihood(PoissonLikelihood):
    name = "Unbinned Clusters"
    columns = ["z", "tsz_signal", "tsz_signal_err", "tile_name"]

    verbose: bool = False
    data: dict = {}
    theorypred: dict = {}
    selfunc: dict = {}
    binning: dict = {}
    YM: dict = {}
    params = {"tenToA0":None, "B0":None, "C0":None, "scatter_sz":None, "bias_sz":None}

    def initialize(self):

        initialize_common(self)

        zmax = self.binning['z']['zmax']

        self.zz = np.arange(0, zmax, 0.01) # redshift bounds should correspond to catalogue
        if self.zz[0] == 0: self.zz[0] = 1e-5

        self.log.info('Number of redshift points for theory calculation = {}.'.format(len(self.zz)))
        self.log.info('Number of mass points for theory calculation = {}.'.format(len(self.lnmarr)))
        self.log.info('Number of y0 points for theory calculation = {}.'.format(len(self.lny)))

        self.catalog = pd.DataFrame(
            {
                "z": self.z_cat,#.byteswap().newbyteorder(),
                "tsz_signal": self.cat_tsz_signal,#.byteswap().newbyteorder(),
                "tsz_signal_err": self.cat_tsz_signal_err,#.byteswap().newbyteorder(),
                "tile_name": self.cat_tile_name#.byteswap().newbyteorder()
            }
        )

        # this is for liklihood computation
        self.zcut = self.binning['exclude_zbin']

        super().initialize()

    def get_requirements(self):
        return get_requirements(self)

    def _get_catalog(self):
        return self.catalog, self.columns

    def Pfunc_inj(self, marr, z, **params):
        if self.theorypred['md_hmf'] != self.theorypred['md_ym']:
            marr_ymmd = convert_masses(self, marr, z)
        else:
            marr_ymmd = marr
        if self.theorypred['md_ym'] != '500c':
            marr_500c = get_m500c(self, marr, z)
        else:
            marr_500c = marr_ymmd

        y0 = get_y0(self, marr_ymmd, z, marr_500c, use_Q=False, Ez_interp=False, **params)
        theta = get_theta(self, marr_500c, z)

        comp = np.zeros_like(theta)
        for i in range(theta.shape[0]):
            comp[i] = self.compThetaInterpolator(theta[i], y0[i]/1e-4, grid=False)
        comp[comp < 0] = 0

        return comp.T


    def _get_n_expected(self, pk_intp, **kwargs):

        start = time.time()

        zz = self.zz
        marr = np.exp(self.lnmarr)
        Ynoise = self.noise

        dVdz = get_dVdz(self, zz, dVdz_interp=False)
        dndlnm = get_dndlnm(self, zz, pk_intp)

        zcut = self.zcut

        if self.selfunc['method'] == 'injection':
            Pfunc = self.Pfunc_inj(marr, zz, **kwargs)
            Pfunc = np.repeat(Pfunc[:,:, np.newaxis], Ynoise.shape[0], axis=2)
        else:
            Pfunc = self.PfuncY(Ynoise, marr, zz, kwargs)

        Ntot = 0
        for index, frac in enumerate(self.skyfracs):
            if zcut > 0:
                Nz = self._get_n_expected_zbinned(zz, dVdz, dndlnm, Pfunc)
                zcut_arr = np.arange(zcut)
                Ntot = np.sum(np.delete(Nz, zcut_arr, 0))
            else:
                Nz = np.trapz(dndlnm * Pfunc[:,:,index], dx=np.diff(self.lnmarr[:, None], axis=0), axis=0)
                Ntot += np.trapz(Nz * dVdz, x=zz) * frac

        self.log.info("Total predicted N = {}".format(Ntot))

        elapsed = time.time() - start
        self.log.info("::: theory N calculation took {:.3f} seconds.".format(elapsed))

        return Ntot

    def _get_n_expected_zbinned(self, zz, dVdz, dndlnm, Pfunc):

        zarr = self.zarr
        nzarr = self.zbins

        Nz = 0
        for index, frac in enumerate(self.skyfracs):
            Nz += np.trapz(dndlnm * dVdz * Pfunc[:,:,index], x=self.lnmarr[:, None], axis=0) * frac

        Nzz = np.zeros(len(zarr))
        for i in range(len(zarr)):
            test = np.abs(zz - nzarr[i])
            i1 = np.argmin(test)
            test = np.abs(zz - nzarr[i+1])
            i2 = np.argmin(test)

            Nzz[i] = np.trapz(Nz[i1:i2+1], x=zz[i1:i2+1])

        # self.log.info("\r Total predicted N = {}".format(Nzz.sum()))
        # for i in range(len(zarr)):
        #     self.log.info('Number of clusters in redshift bin {}: {}.'.format(i, Nzz[i]))
        # self.log.info('------------')

        return Nzz


    def _get_rate_fn(self, pk_intp, **kwargs):

        zarr = self.zz
        marr = np.exp(self.lnmarr)

        dndlnm = get_dndlnm(self, zarr, pk_intp)
        dndlnm_intp = interp2d(zarr, self.lnmarr, dndlnm, kind='cubic', fill_value=0)

        def Prob_per_cluster(z, tsz_signal, tsz_signal_err, tile_name):

            c_z = z
            c_y = tsz_signal * 1e-4
            c_yerr = tsz_signal_err * 1e-4
            c_tile = tile_name

            tile_index = [self.tiles_dwnsmpld[c] for c in c_tile]

            c_z, c_y, c_yerr, tile_index = zip(*sorted(zip(c_z, c_y, c_yerr, tile_index)))
            c_z, c_y, c_yerr, tile_index = np.array(c_z), np.array(c_y), np.array(c_yerr), np.array(tile_index)

            zcut = self.zcut
            if zcut > 0:
                ind = np.where(c_z > self.zbins[zcut])[0]
                c_z, c_y, c_yerr, tile_index = c_z[ind], c_y[ind], c_yerr[ind], tile_index[ind]
                print("::: Excluding clusters of z < {} in likelihood.".format(self.zbins[zcut]))
                print("Total observed N = {}".format(len(c_z)))

            Pfunc_ind = self.Pfunc_per(tile_index, marr, c_z, c_y, c_yerr, kwargs).T
            dn_dlnm = np.squeeze(dndlnm_intp(c_z, self.lnmarr))
            dVdz = get_dVdz(self, c_z, dVdz_interp=True)
            ans = np.trapz(dn_dlnm * Pfunc_ind * dVdz[None,:], x=self.lnmarr[:,None], axis=0)
            #ans = np.trapz(dn_dlnm * Pfunc_ind * dVdz[None,:] * self.skyfracs.sum(), x=self.lnmarr[:,None], axis=0)

            return ans

        return Prob_per_cluster

    def P_Yo(self, tile_index, LgY, marr, z, params):

        if self.theorypred['md_hmf'] != self.theorypred['md_ym']:
            marr_ymmd = convert_masses(self, marr, z)
        else:
            marr_ymmd = marr
        if self.theorypred['md_ym'] != '500c':
            marr_500c = get_m500c(self, marr, z)
        else:
            marr_500c = marr_ymmd

        Ytilde = get_y0(self, marr_ymmd, z, marr_500c, use_Q=True, Ez_interp=True, tile_index=tile_index, **params)

        Ytilde = np.repeat(Ytilde[:,:, np.newaxis], LgY.shape[-1], axis=2)
        LgY = np.repeat(LgY[np.newaxis, :,:], Ytilde.shape[0], axis=0)

        Y = np.exp(LgY)

        numer = -1.0 * (np.log(Y / Ytilde)) ** 2

        ans = (
                1.0 / (params["scatter_sz"] * np.sqrt(2 * np.pi)) *
                np.exp(numer / (2.0 * params["scatter_sz"] ** 2))
        )
        return ans

    def P_Yo_vec(self, LgY, marr, z, params):

        if self.theorypred['md_hmf'] != self.theorypred['md_ym']:
            marr_ymmd = convert_masses(self, marr, z)
        else:
            marr_ymmd = marr
        if self.theorypred['md_ym'] != '500c':
            marr_500c = get_m500c(self, marr, z)
        else:
            marr_500c = marr_ymmd

        Y = np.exp(LgY).T
        Ytilde = get_y0(self, marr_ymmd, z, marr_500c, use_Q=True, Ez_interp=False, **params)

        Y = np.repeat(Y[np.newaxis, :,:,:], Ytilde.shape[0], axis=0)
        Ytilde = np.repeat(Ytilde[:, np.newaxis, :,:], Y.shape[1], axis=1)

        numer = -1.0 * (np.log(Y / Ytilde)) ** 2

        ans = (
                1.0 / (params["scatter_sz"] * np.sqrt(2 * np.pi)) *
                np.exp(numer / (2.0 * params["scatter_sz"] ** 2))
        )
        return ans

    def P_of_gt_SN(self, LgY, marr, z, Ynoise, params):

        if params['scatter_sz'] == 0:

            if self.theorypred['md_hmf'] != self.theorypred['md_ym']:
                marr_ymmd = convert_masses(self, marr, z)
            else:
                marr_ymmd = marr
            if self.theorypred['md_ym'] != '500c':
                marr_500c = get_m500c(self, marr, z)
            else:
                marr_500c = marr_ymmd

            Ytilde = get_y0(self, marr_ymmd, z, marr_500c, use_Q=True, Ez_interp=False, **params)

            qcut = np.outer(np.ones(np.shape(Ytilde)), self.qcut)
            qcut_a = np.reshape(qcut, (Ytilde.shape[0], Ytilde.shape[1], Ytilde.shape[2]))

            Ynoise = np.outer(Ynoise, np.ones(np.shape(Ytilde[0,:,:])))
            Ynoise_a = np.reshape(Ynoise, (Ytilde.shape[0], Ytilde.shape[1], Ytilde.shape[2]))

            if self.selfunc['bias_handler'] == 'theory':
                trueSNR=Ytilde/Ynoise_a
                # bias_pars=self.selfunc['bias_model_params']
                bias_pars={'efold': params['opt_bias_efold'], 'ped': params['opt_bias_ped'], 'norm': params['opt_bias_norm']}
                opt_bias_corr_factor=_opt_bias_func(trueSNR, bias_pars['efold'], bias_pars['ped'], bias_pars['norm'])
                Ytilde=Ytilde*opt_bias_corr_factor

            ans = np.nan_to_num(get_erf(Ytilde, Ynoise_a, qcut_a)).T
            #ans = np.nan_to_num(get_stf(Ytilde, Ynoise_a, qcut_a)).T

        else:

            Y = np.exp(LgY)

            Y_a = np.repeat(Y[:, np.newaxis], np.shape(Ynoise), axis=1)
            Ynoise_a = np.repeat(Ynoise[np.newaxis, :], np.shape(Y), axis=0)

            qcut = np.outer(np.ones(np.shape(Y_a)), self.qcut)
            qcut_a = np.reshape(qcut, (Y_a.shape[0], Y_a.shape[1]))

            if self.selfunc['bias_handler'] == 'theory':
                trueSNR=Y_a/Ynoise_a
                # bias_pars=self.selfunc['bias_model_params']
                bias_pars={'efold': params['opt_bias_efold'], 'ped': params['opt_bias_ped'], 'norm': params['opt_bias_norm']}
                opt_bias_corr_factor=_opt_bias_func(trueSNR, bias_pars['efold'], bias_pars['ped'], bias_pars['norm'])
                Y_a=Y_a*opt_bias_corr_factor

            Yerf = get_erf(Y_a, Ynoise_a, qcut_a)
            #Yerf = get_stf(Y_a, Ynoise_a, qcut_a)

            sig_tr = np.outer(np.ones([marr.shape[0], z.shape[0]]), Yerf)
            sig_thresh = np.reshape(sig_tr, (marr.shape[0], z.shape[0], Yerf.shape[0], Yerf.shape[1]))

            LgYa = np.outer(np.ones([marr.shape[0], z.shape[0]]), LgY)
            LgYa2 = np.reshape(LgYa, (marr.shape[0], z.shape[0], len(LgY)))

            # replace nan with 0's:
            P_Y = np.nan_to_num(self.P_Yo_vec(LgYa2, marr, z, params).T)

            ans = np.trapz(P_Y * sig_thresh, x=LgY, axis=2) #* np.log(10) # why log10?

        return ans

    def PfuncY(self, Ynoise, marr, z, params):
        LgY = self.lny
        P_func = np.outer(marr, np.zeros([len(z)]))
        P_func = self.P_of_gt_SN(LgY, marr, z, Ynoise, params)
        return P_func

    def Y_prob(self, Y_c, LgY, Ynoise):
        Y = np.exp(LgY)
        ans = gaussian(Y, Y_c, Ynoise)
        return ans

    def Pfunc_per(self, tile_index, marr, z, Y_c, Y_err, params):
        if params["scatter_sz"] == 0:
            if self.theorypred['md_hmf'] != self.theorypred['md_ym']:
                marr_ymmd = convert_masses(self, marr, z)
            else:
                marr_ymmd = marr
            if self.theorypred['md_ym'] != '500c':
                marr_500c = get_m500c(self, marr, z)
            else:
                marr_500c = marr_ymmd

            Ytilde = get_y0(self, marr_ymmd, z, marr_500c, use_Q=True, Ez_interp=True, tile_index=tile_index, **params)
            # LgYtilde = np.log(Ytilde)
            # P_Y_sig = np.nan_to_num(self.Y_prob(Y_c[:, None], LgYtilde, Y_err[:, None]))
            # ans = P_Y_sig

            ans = np.nan_to_num(get_erf(Ytilde, Y_err[:, None], self.qcut)) # dN/dz

        else:
            LgY = self.lny
            Y = np.exp(LgY)
            LgYa = np.outer(np.ones(len(marr)), LgY)

            P_Y_sig = np.nan_to_num(get_erf(Y, Y_err[:,None], self.qcut))
            #P_Y_sig = self.Y_prob(Y_c[:, None], LgY, Y_err[:, None])

            P_Y = np.nan_to_num(self.P_Yo(tile_index, LgYa, marr, z, params))
            P_Y_sig = np.repeat(P_Y_sig[:, np.newaxis, :], P_Y.shape[1], axis=1)
            LgY = LgY[None, None, :]
            ans = np.trapz(P_Y * P_Y_sig, LgY, np.diff(LgY), axis=2)

        return ans


def initialize_common(self):
    self.log = logging.getLogger(self.name)
    handler = logging.StreamHandler()
    self.log.addHandler(handler)
    self.log.propagate = False
    if self.verbose:
        self.log.setLevel(logging.INFO)
    else:
        self.log.setLevel(logging.ERROR)
    self.log.info('Initializing clusters.py ' + self.name)

    self.qcut = self.selfunc['SNRcut']
    self.datafile = os.path.abspath(self.data['cat_file'])
    self.selfn_dir = os.path.abspath(self.data['selfn_path'])
    if 'footprint' in self.data.keys():
        self.footprint = self.data['footprint']
        self.log.info('Footprint = {}.'.format(self.footprint))
    else:
        self.footprint = None

    if self.selfunc['method'] == 'SNRbased':
        self.log.info('Running SNR based selection function.')
    elif self.selfunc['method'] == 'injection':
        self.log.info('Running injection based selection function.')
    else:
        print('please choose the method : SNRbased or injection')
        exit(0)

    if self.selfunc['whichQ'] == 'fit':
        self.log.info('Using Qfit data.')
    elif self.selfunc['whichQ'] == 'injection':
        self.log.info('Using averaged Q from source injection.')
        # Qsource only provides the average
    else:
        print('please choose the Q data : Qfit or injection')
        exit(0)

    if self.selfunc['resolution'] == 'full':
        self.log.info('Running completeness with full selection function inputs. No downsampling.')
    elif self.selfunc['resolution'] == 'downsample':
        assert self.selfunc['dwnsmpl_bins'] is not None, 'resolution = downsample but no bin number given. Aborting.'
        self.log.info('Running completeness with down-sampled selection function inputs.')

    cat_tab = atpy.Table().read(self.datafile)
    if self.footprint is not None:
        cat_tab = cat_tab[cat_tab['footprint_{}'.format(self.footprint)] == True]
    zcat = cat_tab['redshift'].data.astype(float)
    qcat = cat_tab['fixed_SNR'].data.astype(float)
    cat_tsz_signal = cat_tab['fixed_y_c'].data.astype(float)
    cat_tsz_signal_err = cat_tab['fixed_err_y_c'].data.astype(float)
    cat_tile_name = np.array(cat_tab['tileName'].data, dtype = str)

    # Optimization bias handler
    if self.selfunc['bias_handler'] not in ['theory', 'catalog']:
        raise NotImplementedError('bias_handler should be either "theory" or "catalog"')
    if self.selfunc['bias_handler'] == 'catalog':
        debiasDOF = self.selfunc['debiasDOF']
        qcat = np.sqrt(np.power(qcat, 2) - debiasDOF)

    # only above given SNR cut
    ind = np.where(qcat >= self.qcut)[0]
    self.z_cat = zcat[ind]
    self.q_cat = qcat[ind]
    self.cat_tsz_signal = cat_tsz_signal[ind]
    self.cat_tsz_signal_err = cat_tsz_signal_err[ind]
    self.cat_tile_name = cat_tile_name[ind]

    self.N_cat = len(self.q_cat)
    self.log.info('Total number of clusters in catalogue = {}.'.format(len(zcat)))
    self.log.info('SNR cut = {}.'.format(self.qcut))
    self.log.info('Number of clusters above the SNR cut = {}.'.format(self.N_cat))
    self.log.info('The lowest redshift = {:.2f}'.format(self.z_cat.min()))
    self.log.info('The highest redshift = {:.2f}'.format(self.z_cat.max()))
    self.log.info('The lowest SNR = {:.2f}.'.format(self.q_cat.min()))
    self.log.info('The highest SNR = {:.2f}.'.format(self.q_cat.max()))

    if self.z_cat.max() > self.binning['z']['zmax']:
        print("Maximum redshift from catalogue is out of given redshift range. Please widen the redshift range for prediction.")
        exit(0)

    # redshift bins for N(z)
    self.zbins = np.arange(self.binning['z']['zmin'], self.binning['z']['zmax'] + self.binning['z']['dz'], self.binning['z']['dz'])
    self.zarr =  0.5*(self.zbins[:-1] + self.zbins[1:])

    self.lnmmin = np.log(self.binning['M']['Mmin'])
    self.lnmmax = np.log(self.binning['M']['Mmax'])
    self.dlnm = self.binning['M']['dlogM']
    self.lnmarr = np.arange(self.lnmmin+(self.dlnm/2.), self.lnmmax, self.dlnm)

    # Ytrue bins if scatter != 0:
    lnymin = -14.  # ln(1e-6) = -13.8
    lnymax = -5.   # ln(1e-2.5) = -5.7
    dlny = 0.2
    lnybins = np.arange(lnymin, lnymax, dlny)
    self.lny = 0.5*(lnybins[:-1] + lnybins[1:])

    # this is to be consist with szcounts.f90
    self.k = np.logspace(-4, np.log10(4), 200, endpoint=False)

    if self.footprint is None:
        self.datafile_rms = self.selfn_dir + os.path.sep + "RMSTab.fits"
    else:
        self.datafile_rms = self.selfn_dir + os.path.sep + "RMSTab_%s.fits" % (self.footprint)
    self.datafile_Q = self.selfn_dir + os.path.sep + "QFit.fits"

    # We need to get rid of the below some how
    self.datafile_tile = self.selfn_dir + os.path.sep + "tileAreas.txt"

    with fits.open(self.datafile_rms) as in_file:
        file_rms = in_file[1].data

    if self.selfunc['resolution'] == 'downsample':

        filename_Q, ext = os.path.splitext(self.datafile_Q)
        datafile_Q_dwsmpld = filename_Q + \
            'dwsmpld_nbins={}'.format(self.selfunc['dwnsmpl_bins']) + '.npz'

        if os.path.exists(datafile_Q_dwsmpld):
            Qfile = np.load(datafile_Q_dwsmpld)
            self.Q = Qfile['Q_dwsmpld']
            self.tt500 = Qfile['tt500']
            self.log.info("Down-sampled Q funcs exists. Number of Q funcs = {}.".format(len(self.Q[0])))

        else:
            self.log.info("Reading in full Q function.")

            # Old  - we want to ultimately remove
            # tile_info = np.genfromtxt(self.datafile_tile, dtype=str)
            # tile_area0 = tile_info[:, 1]
            # zero_index = np.where(tile_area0 == '0.000000')[0]
            # tile_area = np.delete(tile_info, zero_index, 0)
            # tile_name = tile_area[:, 0]

            tile_name = np.unique(file_rms['tileName'])
            QFit = nm.signals.QFit(QSource=self.selfunc['whichQ'], selFnDir=self.selfn_dir,
                                   tileNames = tile_name)
            Nt = len(tile_name)
            self.log.info("Number of tiles = {}.".format(Nt))
            self.tname = np.array(file_rms['tileName'], dtype = str) # Avoids potential chararray weirdness

            with fits.open(self.datafile_Q) as hdulist:
                data = hdulist[1].data
                tt500 = data.field("theta500Arcmin")

            # reading in all Q functions
            allQ = np.zeros((len(tt500), Nt))
            for i in range(Nt):
                allQ[:, i] = QFit.getQ(tt500, tileName=tile_name[i])
            assert len(tt500) == len(allQ[:, 0])
            self.tt500 = tt500
            self.Q = allQ

        filename_rms, ext = os.path.splitext(self.datafile_rms)
        filename_tile, ext = os.path.splitext(self.datafile_tile)
        datafile_rms_dwsmpld = filename_rms + \
            'dwsmpld_nbins={}'.format(self.selfunc['dwnsmpl_bins']) + '.npz'
        datafile_tiles_dwsmpld = filename_tile + \
            'dwsmpld_nbins={}'.format(self.selfunc['dwnsmpl_bins']) + '.npy'

        if os.path.exists(datafile_rms_dwsmpld):
            rms = np.load(datafile_rms_dwsmpld)
            self.noise = rms['noise']
            self.skyfracs = rms['skyfracs']
            self.log.info("Down-sampled RMS table exists. Number of RMS bins = {}.".format(self.skyfracs.size))

            self.tiles_dwnsmpld = np.load(datafile_tiles_dwsmpld, allow_pickle='TRUE').item()

        else:

            self.log.info("Reading in full RMS table.")

            self.noise = file_rms['y0RMS']
            self.skyfracs = file_rms['areaDeg2'] * np.deg2rad(1.) ** 2
            self.log.info("Number of RMS values = {}.".format(self.skyfracs.size))
            self.log.info("Down-sampling RMS and Q function using {} bins.".format(self.selfunc['dwnsmpl_bins']))
            binned_stat = stats.binned_statistic(self.noise, self.skyfracs, statistic='sum',
                                                       bins=self.selfunc['dwnsmpl_bins'])
            binned_area = binned_stat[0]
            binned_rms_edges = binned_stat[1]

            bin_ind = np.digitize(self.noise, binned_rms_edges)
            tiledict = dict(zip(tile_name, np.arange(tile_name.shape[0])))

            Qdwnsmpld = np.zeros((self.Q.shape[0], self.selfunc['dwnsmpl_bins']))
            tiles_dwnsmpld = {}

            for i in range(self.selfunc['dwnsmpl_bins']):
                tempind = np.where(bin_ind == i + 1)[0]
                if len(tempind) == 0:
                    #self.log.info('Found empty bin.')
                    Qdwnsmpld[:, i] = np.zeros(self.Q.shape[0])
                else:
                    #print('dowsampled rms bin ',i)
                    temparea = self.skyfracs[tempind]
                    #print('areas of tiles in bin',temparea)
                    temptiles = self.tname[tempind]
                    #print('names of tiles in bin',temptiles)
                    for t in temptiles:
                        tiles_dwnsmpld[t] = i

                    test = [tiledict[key] for key in temptiles]
                    Qdwnsmpld[:, i] = np.average(self.Q[:, test], axis=1, weights=temparea)

            self.noise = 0.5*(binned_rms_edges[:-1] + binned_rms_edges[1:])
            self.skyfracs = binned_area
            self.Q = Qdwnsmpld
            self.tiles_dwnsmpld = tiles_dwnsmpld

            if 'override_noise' in self.selfunc.keys():
                self.log.info("Overriden noise")
                self.noise[:]=self.selfunc['override_noise']

            if 'force_Q_equals_1' in self.selfunc.keys() and self.selfunc['force_Q_equals_1'] == True:
                self.log.info("Forced Q == 1")
                self.Q[:] = 1.0


            self.log.info("Number of down-sampled RMS = {}.".format(self.skyfracs.size))
            self.log.info("Number of down-sampled Q funcs = {}.".format(len(self.Q[0])))

            assert self.noise.shape[0] == self.skyfracs.shape[0] and self.noise.shape[0] == self.Q.shape[1]

            if self.selfunc['save_dwsmpld']:
                np.savez(datafile_Q_dwsmpld, Q_dwsmpld=Qdwnsmpld, tt500=self.tt500)
                np.savez(datafile_rms_dwsmpld, noise=self.noise, skyfracs=self.skyfracs)
                np.save(datafile_tiles_dwsmpld, self.tiles_dwnsmpld)


    elif self.selfunc['resolution'] == 'full':

        self.log.info('Reading in full Q function.')
        tile_info = np.genfromtxt(os.path.join(self.data_directory, self.data['tile_file']), dtype=str)

        # removing tiles with zero areas
        tile_area0 = tile_info[:, 1]
        zero_index = np.where(tile_area0 == '0.000000')[0]
        tile_area = np.delete(tile_info, zero_index, 0)

        tile_name = tile_area[:, 0]
        QFit = nm.signals.QFit(QFitFileName=os.path.join(self.data_directory, self.datafile_Q),
                                   tileNames=tile_name, QSource=self.selfunc['whichQ'], selFnDir=self.data_directory+'/selFn')
        Nt = len(tile_name)
        self.log.info("Number of tiles = {}.".format(Nt))

        hdulist = fits.open(os.path.join(self.data_directory, self.datafile_Q))
        data = hdulist[1].data
        tt500 = data.field("theta500Arcmin")

        # reading in all Q functions
        allQ = np.zeros((len(tt500), Nt))
        for i in range(Nt):
            allQ[:, i] = QFit.getQ(tt500, tileName=tile_name[i])
        assert len(tt500) == len(allQ[:, 0])
        self.tt500 = tt500
        self.Q = allQ

        # when using full Q functions, noise values should be downsampled
        # in a current setting, the number of noise values has to be same as the number of Q funcs
        # hence the number of tiles - they are averaged by each tile

        self.log.info("Reading in full RMS table.")
        self.log.info("Number of RMS values = {}.".format(len(file_rms['y0RMS'])))
        self.log.info("Down-sampling RMS using {} bins.".format(Nt))

        self.tname = file_rms['tileName']

        noisebyTile = {}
        areabyTile = {}
        for t in tile_name:
            tileTab = file_rms[self.tname == t]
            areaWeights = tileTab['areaDeg2'] / tileTab['areaDeg2'].sum()
            noisebyTile[t] = np.average(tileTab['y0RMS'], weights=areaWeights)
            areabyTile[t] = tileTab['areaDeg2'].sum()

        self.noise = np.array([noisebyTile[t] for t in tile_name])
        self.skyfracs = np.array([areabyTile[t] for t in tile_name]) * np.deg2rad(1.)**2

        self.log.info("Number of down-sampled RMS = {}.".format(self.skyfracs.size))
        self.log.info("Number of Q funcs = {}.".format(len(self.Q[0])))

        assert self.noise.shape[0] == self.skyfracs.shape[0] and self.noise.shape[0] == self.Q.shape[1]


        # choosing tile ----------------------------------------------------

        if self.selfunc['tiletest'] == True:

            tile_index = 0
            # tile_index = slice(120, 123, None)
            print('Name of tile : ', tile_name[tile_index])

            self.Q = self.Q[:, tile_index]
            self.Q = self.Q[:, None]
            self.noise = np.array([self.noise[tile_index]])
            self.skyfracs = np.array([self.skyfracs[tile_index]])
            # self.noise = self.noise[tile_index]
            # self.skyfracs = self.skyfracs[tile_index]

        #-------------------------------------------------------------------



        # # fiddling Q fit using injection Q ---------------------------------
        #
        # if self.selfunc['Qtest'] == True:
        #
        #     injQFit = nm.signals.QFit(QFitFileName=os.path.join(self.data_directory, self.datafile_Q),
        #                             tileNames=tile_name, QSource='injection', selFnDir=self.data_directory+'/selFn')
        #
        #     injQ = np.zeros(len(tt500))
        #     injQ = injQFit.getQ(tt500, tileName=tile_area[:, 0][0])
        #
        #     meanQ = np.average(allQ, axis=1)
        #     fac = injQ / meanQ
        #     fac_arr = np.repeat(fac[:, np.newaxis], allQ.shape[1], axis=1)
        #
        #     self.Q = allQ * fac_arr
        #
        # #-------------------------------------------------------------------

    if self.selfunc['method'] == 'injection':

        try:
            self.compThetaInterpolator = selfunc.get_completess_inj_theta_y(self.data_directory, self.qcut, self.qbins)
        except:
            self.compThetaInterpolator = selfunc.get_completess_inj_theta_y_unb(self.data_directory, self.qcut)


    self.log.info('Entire survey area = {} deg2.'.format(self.skyfracs.sum()/(np.deg2rad(1.)**2.)))


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

def get_Ez(both, zarr, Ez_interp):
    if Ez_interp: # interpolation is needed for Pfunc_per in unbinned
        Ez = interp1d(both.zz, both.theory.get_Hubble(both.zz) / both.theory.get_param("H0"))
        return Ez(zarr)
    else:
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

def get_dVdz(both, zarr, dVdz_interp):
    """dV/dzdOmega"""

    if dVdz_interp:
        Da_intp = interp1d(both.zz, both.theory.get_angular_diameter_distance(both.zz))
        DA_z = Da_intp(zarr)
        H_intp = interp1d(both.zz, both.theory.get_Hubble(both.zz))
        H_z = H_intp(zarr)
    else:
        DA_z = both.theory.get_angular_diameter_distance(zarr)
        H_z = both.theory.get_Hubble(zarr)

    dV_dz = (
        DA_z**2
        * (1.0 + zarr) ** 2
        / (H_z / C_KM_S)
    )
    h = both.theory.get_param("H0") / 100.0
    return dV_dz*h**3

def get_dndlnm(self, z, pk_intp):

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

        k = self.k #np.logspace(-4, np.log10(4), 200, endpoint=False)
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
            return (0.75*M/np.pi/rhom0)**(1./3.)

        def win(x):
            return 3.*(np.sin(x) - x*np.cos(x))/(x**3.)

        def win_prime(x):
            return 3.*np.sin(x)/(x**2.) - 9.*(np.sin(x) - x*np.cos(x))/(x**4.)

        def sigma_sq(R, k):
            integral = np.zeros((len(k), len(marr), len(z)))
            for i in range(k.size):
                integral[i,:,:] = np.array((k[i]**2.)*pks[:,i]*(win(k[i]*R)**2.))
            return integrate.simps(integral, k, axis=0)/(2.*np.pi**2.)

        def sigma_sq_prime(R, k):
            # this is derivative of sigmaR squared
            # so 2 * sigmaR * dsigmaR/dR
            integral = np.zeros((len(k), len(marr), len(z)))
            for i in range(k.size):
                integral[i,:,:] = np.array((k[i]**2.)*pks[:,i]*2.*k[i]*win(k[i]*R)*win_prime(k[i]*R))
            return integrate.simps(integral, k, axis=0)/(2.*np.pi**2.)

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
        return hmf_internal

    elif self.theorypred['massfunc_mode'] == 'ccl':
        # First, gather all the necessary ingredients for the number counts
        mf = self.theory.get_nc_data()['HMF']
        cosmo = self.theory.get_CCL()['cosmo']

        h = self.theory.get_param("H0") / 100.0
        a = 1./(1+z)
        marr = np.exp(marr)
        dn_dlog10M = np.array([mf(cosmo, marr/h, ai) for ai in a])
        # For consistency with internal mass function computation
        dn_dlog10M /= h**3 * np.log(10.)

        return dn_dlog10M.T

    # elif self.theorypred['massfunc_mode'] == 'class_sz':
        # return self.get_dndlnM_at_z_and_M(z,marr)


def get_erf(y, noise, cut):
    #arg = (y - cut*noise)/np.sqrt(2.)/noise
    arg = (y/noise - cut)/np.sqrt(2.)
    erfc = (special.erf(arg) + 1.)/2.
    return erfc

def get_stf(y, noise, qcut):
    ans = y * 0.0
    ans[y - qcut*noise > 0] = 1.0
    return ans

def get_erf_diff(y, noise, qmin, qmax, qcut):
    arg1 = (y/noise - qmax)/np.sqrt(2.)
    if qmin > qcut:
        qlim = qmin
    else:
        qlim = qcut
    arg2 = (y/noise - qlim)/np.sqrt(2.)
    erf_compl = (special.erf(arg2) - special.erf(arg1)) / 2.
    # HACK: 100% completeness above S/N limit, for noiseless data vector - doesn't actually help
    # erf_compl[erf_compl > 0]=1.0
    return erf_compl

def get_stf_diff(y, noise, qmin, qmax, qcut):
    if qmin > qcut:
        qmin = qmin
    else:
        qmin = qcut
    ans = y * 0.0
    ans[(y - qmin*noise > 0) & (y - qmax*noise < 0)] = 1.0
    return ans

def get_erf_prod(y, noise, qmin, qmax, qcut, k, Nq):

    arg0 = get_erf(y, noise, qcut)
    arg1 = get_erf(y, noise, qmin)
    arg2 = 1. - get_erf(y, noise, qmax)

    if k == 0: arg1 = 1
    if k == Nq-1: arg2 = 1

    return arg0 * arg1 * arg2

def get_stf_prod(y, noise, qmin, qmax, qcut, k, Nq):

    arg0 = get_stf(y, noise, qcut)
    arg1 = get_stf(y, noise, qmin)
    arg2 = 1. - get_stf(y, noise, qmax)

    if k == 0: arg1 = 1
    if k == Nq-1: arg2 = 1

    return arg0 * arg1 * arg2

def gaussian(xx, mu, sig, noNorm=False):
    if noNorm:
        return np.exp(-1.0 * (xx - mu) ** 2 / (2.0 * sig ** 2.0))
    else:
        return 1.0 / (sig * np.sqrt(2 * np.pi)) \
                            * np.exp(-1.0 * (xx - mu) ** 2 / (2.0 * sig ** 2.0))


def convert_masses(both, marr, zz):

    h = both.theory.get_param("H0") / 100.0
    if both.theorypred['choose_theory'] == 'CCL':
        mf_data = both.theory.get_nc_data()
        md_hmf = mf_data['md']

        if both.theorypred['md_ym'] == '200m':
            md_ym = ccl.halos.MassDef(200, 'matter')
        elif both.theorypred['md_ym'] == '200c':
            md_ym = ccl.halos.MassDef(200, 'critical')
        elif both.theorypred['md_ym'] == '500c':
            md_ym = ccl.halos.MassDef(500, 'critical')
        else:
            raise NotImplementedError('Only md_hmf = 200m, 200c and 500c currently supported.')
        cosmo = both.theory.get_CCL()['cosmo']
        a = 1. / (1. + zz)
        mass_trans=ccl.halos.mass_translator(mass_in = md_hmf, mass_out = md_ym, concentration = 'Bhattacharya13')
        marr_ymmd = np.array([mass_trans(cosmo, marr / h, ai) for ai in a]) * h
    else:
        if both.theorypred['md_hmf'] == '200m' and both.theorypred['md_ym'] == '500c':
            marr_ymmd = both._get_M500c_from_M200m(marr, zz).T
        else:
            raise NotImplementedError()
    return marr_ymmd

def get_m500c(both, marr, zz):

    h = both.theory.get_param("H0") / 100.0
    mf_data = both.theory.get_nc_data()
    md_hmf = mf_data['md']
    md_500c = ccl.halos.MassDef(500, 'critical')
    cosmo = both.theory.get_CCL()['cosmo']
    a = 1. / (1. + zz)

    mass_trans=ccl.halos.mass_translator(mass_in = md_hmf, mass_out = md_500c, concentration = 'Bhattacharya13')
    if a.ndim == 1:
        marr_500c = np.array([mass_trans(cosmo, marr/h, ai) for ai in a]) * h
    else:
        marr_500c = mass_trans(cosmo, marr/h, a) * h

    return marr_500c

def get_splQ(self, theta, tile_index=None):

    if self.selfunc['whichQ'] == 'injection':
        # this is because injection Q is survey-wide averaged for now

        tck = interpolate.splrep(self.tt500, self.Q[:,0])
        newQ0 = interpolate.splev(theta, tck)
        newQ = np.repeat(newQ0[np.newaxis,...], self.Q.shape[1], axis=0)

        if tile_index is not None: # for faster rate_fn in unbinned

            chosenQ = np.zeros((newQ.shape[1], newQ.shape[2]))
            for i in range(len(tile_index)):
                chosenQ[i, :] = newQ[tile_index[i], i, :]
            newQ = chosenQ

    else:
        newQ = []
        for i in range(len(self.Q[0])):
            tck = interpolate.splrep(self.tt500, self.Q[:, i])
            newQ.append(interpolate.splev(theta, tck))

        if tile_index is not None: # for faster rate_fn in unbinned

            newQ = np.array(newQ)

            chosenQ = np.zeros((newQ.shape[1], newQ.shape[2]))
            for i in range(len(tile_index)):
                chosenQ[i, :] = newQ[tile_index[i], i, :]
            newQ = chosenQ

    return np.asarray(np.abs(newQ))

def get_theta(self, mass_500c, z, Ez=None, Ez_interp=False):

    thetastar = 6.997
    alpha_theta = 1. / 3.
    H0 = self.theory.get_param("H0")
    h = H0/100.0

    if Ez is None:
        Ez = get_Ez(self, z, Ez_interp)
        Ez = Ez[:, None]

    DAz_interp = interp1d(self.zz , self.theory.get_angular_diameter_distance(self.zz) * h)
    DAz = DAz_interp(z)
    try:
        DAz = DAz[:, None]
    except:
        DAz = DAz
    ttstar = thetastar * (H0 / 70.) ** (-2. / 3.)

    return ttstar * (mass_500c / MPIVOT_THETA / h) ** alpha_theta * Ez ** (-2. / 3.) * (100. * DAz / 500 / H0) ** (-1.)

# y-m scaling relation for completeness
def get_y0(self, mass, z, mass_500c, use_Q=True, Ez_interp=False, tile_index=None, **params):

    A0 = params["tenToA0"]
    B0 = params["B0"]
    C0 = params["C0"]
    bias = params["bias_sz"]

    Ez = get_Ez(self, z, Ez_interp)
    try:
        Ez = Ez[:, None]
    except:
        Ez = Ez

    h = self.theory.get_param("H0") / 100.0

    mb = mass * bias
    mb_500c = mass_500c * bias

    Mpivot = self.YM['Mpivot'] * h  # convert to Msun/h.

    def rel(m):
        if self.theorypred['rel_correction']:
            t = -0.008488*(mm*Ez)**(-0.585)
            res = 1.+ 3.79*t - 28.2*(t**2.)
        else:
            res = 1.
        return res

    if use_Q is True:
        theta = get_theta(self, mb_500c, z, Ez)
        splQ = get_splQ(self, theta, tile_index)
    else:
        splQ = 1.

    y0 = A0 * (Ez ** 2.) * (mb / Mpivot) ** (1. + B0) * splQ
    y0[y0 <= 0] = 1e-9

    return y0
