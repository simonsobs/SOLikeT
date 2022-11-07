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
from scipy import special
import math as m

import pyccl as ccl
#from classy_sz import Class # TBD: change this import as optional

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
        # redshift bins for N(z)
        self.zbins = np.arange(self.binning['z']['zmin'], self.binning['z']['zmax'] + self.binning['z']['dz'], self.binning['z']['dz'])
        self.zarr =  0.5*(self.zbins[:-1] + self.zbins[1:])
        self.log.info("Number of redshift bins = {}.".format(len(self.zarr)))

        # constant binning in log10
        qbins = np.arange(self.binning['q']['log10qmin'], self.binning['q']['log10qmax']+self.binning['q']['dlog10q'], self.binning['q']['dlog10q'])
        self.qbins = 10**qbins
        self.qarr = 10**(0.5*(qbins[:-1] + qbins[1:]))
        self.Nq = int((self.binning['q']['log10qmax'] - self.binning['q']['log10qmin'])/self.binning['q']['dlog10q']) + 1

        # Ytrue bins if scatter != 0:
        lnymin = -25.  # ln(1e-10) = -23
        lnymax = 0.  # ln(1e-2) = -4.6
        dlny = 0.05
        lnybins = np.arange(lnymin, lnymax, dlny)
        self.lny = 0.5*(lnybins[:-1] + lnybins[1:])

        initialize_common(self)

        if self.theorypred['choose_dim'] == '2D':
            self.log.info('2D likelihood as a function of redshift and signal-to-noise.')
        else:
            self.log.info('1D likelihood as a function of redshift.')

        delNcat, _ = np.histogram(self.z_cat, bins=self.zbins)
        self.delNcat = self.zarr, delNcat


        if self.theorypred['choose_dim'] == "2D":
            self.log.info('Number of SNR bins = {}.'.format(self.Nq))
            self.log.info('Edges of SNR bins = {}.'.format(self.qbins))

        delN2Dcat, _, _ = np.histogram2d(self.z_cat, self.q_cat, bins=[self.zbins, self.qbins])
        self.delN2Dcat = self.zarr, self.qarr, delN2Dcat


        # finner binning for low redshift
        minz = self.zarr[0]
        maxz = self.zarr[-1]
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
        self.log.info('Number of redshift points for theory calculation {}.'.format(len(self.zz)))

        super().initialize()

    def get_requirements(self):
        return get_requirements(self)

    def _get_hres_z(self, zi):
        # bins in redshifts are defined with higher resolution for low redshift
        hr = 0.2
        if zi < hr :
            dzi = 1e-2 #1e-3
        elif zi >= hr and zi <=1.:
            dzi = 5e-2 #1e-3
        else:
            dzi = 5e-2 #1e-3 #self.binning['z']['dz']
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

        h = self.theory.get_param("H0") / 100.0

        dndlnm = get_dndlnm(self,zz, pk_intp, **params_values_dict)
        dVdzdO = get_dVdz(self,zz)
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

        if self.selfunc['mode'] != 'injection':
            y0 = _get_y0(self,marr_ymmd, zz, marr_500c, **params_values_dict)
        else:
            y0 = None

        cc = np.array([self._get_completeness2D(marr, zz, y0, kk, marr_500c, **params_values_dict) for kk in range(Nq)])

        nzarr = self.zbins
        delN2D = np.zeros((len(zarr), Nq))

        # integrate over mass
        dndzz = np.trapz(intgr[None,:,:]*cc, dx=self.dlnm, axis=2)

        # integrate over fine z bins and sum over in larger z bins
        for i in range(len(zarr)):

            test = np.abs(zz - nzarr[i])
            i1 = np.argmin(test)
            test = np.abs(zz - nzarr[i+1])
            i2 = np.argmin(test)

            delN2D[i,:] = np.trapz(dndzz[:,i1:i2+1], x=zz[i1:i2+1]).T

        self.log.info("\r Total predicted 2D N = {}".format(delN2D.sum()))

        for i in range(len(zarr)):
            self.log.info('Number of clusters in redshift bin {}: {}.'.format(i, delN2D[i,:].sum()))
        self.log.info('------------')
        for kk in range(Nq):
            self.log.info('Number of clusters in snr bin {}: {}.'.format(kk, delN2D[:,kk].sum()))
        self.log.info("Total predicted 2D N = {}.".format(delN2D.sum()))

        return delN2D

    def get_completeness2D_inj(self, mass, z, mass_500c, qbin, **params_values_dict):

        scatter = params_values_dict["scatter_sz"]

        y0 = _get_y0(self,mass, z, mass_500c, use_Q=False, **params_values_dict)
        theta = _theta(self,mass_500c, z)

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


    # completeness 2D
    def _get_completeness2D(self, marr, zarr, y0, qbin, marr_500c=None, **params_values_dict):
        if self.selfunc['mode'] != 'injection':
            scatter = params_values_dict["scatter_sz"]
            noise = self.noise
            qcut = self.qcut
            skyfracs = self.skyfracs/self.skyfracs.sum()
            Npatches = len(skyfracs)

            if not self.selfunc['average_Q']:
                if self.selfunc['Qmode'] == 'downsample':
                    tile_list = np.arange(noise.shape[0])+1
                elif self.selfunc['Qmode'] == 'full':
                    tile_list = self.tile_list
            else:
                tile_list = None

            Nq = self.Nq
            qbins = self.qbins

            compl_mode = self.theorypred['compl_mode']
            tile = tile_list

            # can I do something about loop for SNR?
            kk = qbin
            qmin = qbins[kk]
            qmax = qbins[kk+1]

            if scatter == 0.:

                if self.selfunc['average_Q']:
                    if compl_mode == 'erf_prod':
                        if kk == 0:
                            arg = get_erf(y0, noise, qcut)*(1. - get_erf(y0, noise, qmax))
                        elif kk == Nq:
                            arg = get_erf(y0, noise, qcut)*get_erf(y0, noise, qmin)
                        else:
                            arg = get_erf(y0, noise, qcut)*get_erf(y0, noise, qmin)*(1. - get_erf(y0, noise, qmax))
                    elif compl_mode == 'erf_diff':
                        arg = get_erf_compl(y0, qmin, qmax, noise, qcut)

                    comp = arg.T

                else:
                    arg = []
                    for i in range(len(skyfracs)):
                        if compl_mode == 'erf_prod':
                            if kk == 0:
                                arg.append(get_erf(y0[int(tile[i])-1,:,:], noise[i], qcut)*(1. - get_erf(y0[int(tile[i])-1,:,:], noise[i], qmax)))
                            elif kk == Nq:
                                arg.append(get_erf(y0[int(tile[i])-1,:,:], noise[i], qcut)*get_erf(y0[int(tile[i])-1,:,:], noise[i], qmin))
                            else:
                                arg.append(get_erf(y0[int(tile[i])-1,:,:], noise[i], qcut)*get_erf(y0[int(tile[i])-1,:,:], noise[i], qmin)*(1. - get_erf(y0[int(tile[i])-1,:,:], noise[i], qmax)))
                        elif compl_mode == 'erf_diff':
                            arg.append(get_erf_compl(y0[int(tile[i])-1,:,:], qmin, qmax, noise[i], qcut))

                    comp = np.einsum('ijk,i->jk', arg, skyfracs)

            else:

                lnyy = self.lny
                yy0 = np.exp(lnyy)

                mu = np.log(y0)
                fac = 1./np.sqrt(2.*np.pi*scatter**2)

                if self.selfunc['average_Q']:
                    if compl_mode == 'erf_prod':
                        if kk == 0:
                            arg = get_erf(yy0, noise, qcut)*(1. - get_erf(yy0, noise, qmax))
                        elif kk == Nq-1:
                            arg = get_erf(yy0, noise, qcut)*get_erf(yy0, noise, qmin)
                        else:
                            arg = get_erf(yy0, noise, qcut)*get_erf(yy0, noise, qmin)*(1. - get_erf(yy0, noise, qmax))
                    elif compl_mode == 'erf_diff':
                        arg = get_erf_compl(yy0, qmin, qmax, noise, qcut)

                    arg0 = (lnyy[:,None,None] - mu)/(np.sqrt(2.)*scatter)
                    args = fac*np.exp(-arg0**2.)*arg[:,None,None]
                    comp = np.trapz(args, x=lnyy, axis=0).T

                else:
                    comp = 0.

                    for i in range(len(skyfracs)):
                        if compl_mode == 'erf_prod':
                            if kk == 0:
                                arg = get_erf(yy0, noise[i], qcut)*(1. - get_erf(yy0, noise[i], qmax))
                            elif kk == Nq-1:
                                arg = get_erf(yy0, noise[i], qcut)*get_erf(yy0, noise[i], qmin)
                            else:
                                arg = get_erf(yy0, noise[i], qcut)*get_erf(yy0, noise[i], qmin)*(1. - get_erf(yy0, noise[i], qmax))
                        elif compl_mode == 'erf_diff':
                            arg = get_erf_compl(yy0, qmin, qmax, noise[i], qcut)

                        cc = arg * skyfracs[i]
                        arg0 = (lnyy[:,None,None] - mu[int(tile[i])-1,:,:])/(np.sqrt(2.)*scatter)
                        args = fac*np.exp(-arg0**2.)*cc[:,None,None]
                        comp += np.trapz(args, x=lnyy, axis=0)


            comp[comp < 0.] = 0.
            comp[comp > 1.] = 1.
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

        #self.qbins = None
        #self.tiles_dwnsmpld = None

        initialize_common(self)
        self.LgY = np.arange(-6, -2.5, 0.01) # for integration over y when scatter != 0
        self.zz = np.arange(0, 8, 0.05) # redshift bounds should correspond to catalogue
        super().initialize()

    def get_requirements(self):
        return get_requirements(self)


    def _get_dndlnm(self,z, pk_intp, **kwargs):
        return get_dndlnm(self,z, pk_intp, **kwargs)

    def _get_n_expected(self, pk_intp,**kwargs):
        dVdz = get_dVdz(self,self.zz)
        dndlnm = get_dndlnm(self,self.zz, pk_intp, **kwargs)

        Ntot = 0
        rms_index = 0
        marr = np.exp(self.lnmarr)
        for Yt, frac in zip(self.noise, self.skyfracs):
            Pfunc = self.PfuncY(rms_index, Yt, marr, self.zz, kwargs) # dim (m,z)
            Pfunc = Pfunc.T #####

            N_z = np.trapz(
                dndlnm * Pfunc, dx=np.diff(self.lnmarr[:,None], axis=0), axis=0
            ) # dim (z)

            Np = (
                np.trapz(N_z * dVdz, x=self.zz)
                * frac
            )
            Ntot += Np
            rms_index += 1
        self.log.info("\r Total predicted N = {}".format(Ntot))
        return Ntot

    def P_Yo(self, rms_bin_index, LgY, marr, z, param_vals):

        marr = np.outer(marr, np.ones(len(LgY[0, :])))
        # Mass conversion needed!
        mass_500c = marr
        y0_new = _get_y0(self,marr, z, mass_500c, use_Q=True, **param_vals)
        y0_new = y0_new[rms_bin_index]

        Ytilde = y0_new
        Y = 10 ** LgY

        numer = -1.0 * (np.log(Y / Ytilde)) ** 2
        ans = (
                1.0 / (param_vals["scatter_sz"] * np.sqrt(2 * np.pi)) *
                np.exp(numer / (2.0 * param_vals["scatter_sz"] ** 2))
        )
        return ans

    def P_Yo_vec(self, rms_index, LgY, marr, z, param_vals):
        # Mass conversion
        if self.theorypred['md_hmf'] != self.theorypred['md_ym']:
            marr_ymmd = convert_masses(self, marr, z)
        else:
            marr_ymmd = marr
        if self.theorypred['md_ym'] != '500c':
            marr_500c = get_m500c(self, marr, z)
        else:
            marr_500c = marr_ymmd

        y0_new = _get_y0(self, marr_ymmd, z, marr_500c, use_Q=True, **param_vals)
        y0_new = y0_new[rms_index]

        Y = 10 ** LgY
        Ytilde = np.repeat(y0_new[:, :, np.newaxis], LgY.shape[2], axis=2)

        numer = -1.0 * (np.log(Y/ Ytilde)) ** 2 #####

        ans = (
                1.0 / (param_vals["scatter_sz"] * np.sqrt(2 * np.pi)) *
                np.exp(numer / (2.0 * param_vals["scatter_sz"] ** 2))
        )
        return ans

    def Y_erf(self, Y, Ynoise):
        ans = Y * 0.0
        ans[Y - self.qcut * Ynoise > 0] = 1.0
        return ans

    def P_of_gt_SN(self, rms_index, LgY, marr, zz, Ynoise, param_vals):
        if param_vals['scatter_sz'] != 0:
            Y = 10 ** LgY

            Yerf = self.Y_erf(Y, Ynoise) # array of size dim Y
            sig_tr = np.outer(np.ones([marr.shape[0], # (dim mass)
                                        zz.shape[0]]), # (dim z)
                                        Yerf )

            sig_thresh = np.reshape(sig_tr,
                                    (marr.shape[0], zz.shape[0], len(Yerf)))

            LgYa = np.outer(np.ones([marr.shape[0], zz.shape[0]]), LgY)
            LgYa2 = np.reshape(LgYa, (marr.shape[0], zz.shape[0], len(LgY)))

            # replace nan with 0's:
            P_Y = np.nan_to_num(self.P_Yo_vec(rms_index,LgYa2, marr, zz, param_vals))
            ans = np.trapz(P_Y * sig_thresh, x=LgY, axis=2) * np.log(10) # why log10? #####

        else:
            # Mass conversion
            if self.theorypred['md_hmf'] != self.theorypred['md_ym']:
                marr_ymmd = convert_masses(self, marr, zz)
            else:
                marr_ymmd = marr
            if self.theorypred['md_ym'] != '500c':
                marr_500c = get_m500c(self, marr, zz)
            else:
                marr_500c = marr_ymmd
            y0_new = _get_y0(self, marr_ymmd, zz, marr_500c, use_Q=True, **param_vals)
            y0_new = y0_new[rms_index]

            ans = y0_new * 0.0
            ans[y0_new - self.qcut * self.noise[rms_index] > 0] = 1.0 #?
            ans = np.nan_to_num(ans)

        return ans

    def PfuncY(self, rms_index, YNoise, marr, z_arr, param_vals):
        LgY = self.LgY
        P_func = np.outer(marr, np.zeros([len(z_arr)]))
        # marr = np.outer(marr, np.ones([len(z_arr)]))
        P_func = self.P_of_gt_SN(rms_index, LgY, marr, z_arr, YNoise, param_vals)
        return P_func

    def Y_prob(self, Y_c, LgY, YNoise):
        Y = 10 ** (LgY)

        ans = gaussian(Y, Y_c, YNoise)
        return ans

    def Pfunc_per(self, rms_bin_index, marr, zz, Y_c, Y_err, param_vals):
        if param_vals["scatter_sz"] != 0:
            LgY = self.LgY
            LgYa = np.outer(np.ones(len(marr)), LgY)
            P_Y_sig = self.Y_prob(Y_c, LgY, Y_err)
            P_Y = np.nan_to_num(self.P_Yo(rms_bin_index, LgYa, marr, zz, param_vals))
            ans = np.trapz(P_Y * P_Y_sig, LgY, np.diff(LgY), axis=1)
        else:
            # mass conversion needed!
            mass_500c = marr
            y0_new = _get_y0(self,marr, zz, mass_500c, use_Q=True, **param_vals)
            y0_new = y0_new[rms_bin_index]
            LgY = np.log10(y0_new)
            P_Y_sig = np.nan_to_num(self.Y_prob(Y_c, LgY, Y_err))
            ans = P_Y_sig

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
        # SNR cut
        self.qcut = self.selfunc['SNRcut']

        self.datafile = self.data['cat_file']
        self.data_directory = self.data['data_path']

        if self.selfunc['Qmode'] == 'full':
            self.log.info('Running Q-fit completeness with full analysis. No downsampling.')
        elif self.selfunc['Qmode'] == 'downsample':
            assert self.selfunc['dwnsmpl_bins'] is not None, 'mode = downsample but no bin number given. Aborting.'
            self.log.info('Running Q-fit completeness with downsampling selection function inputs.')

        if self.selfunc['mode'] == 'injection':
            self.log.info('Running injection based selection function. Currently using one average Q function.')

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
        self.q_cat = qcat[ind]
        # SPT-style SNR bias correction
        debiasDOF = 0
        self.q_cat = np.sqrt(np.power(self.q_cat, 2) - debiasDOF)
        self.cat_tsz_signal = cat_tsz_signal[ind]
        self.cat_tsz_signal_err = cat_tsz_signal_err[ind]
        self.cat_tile_name = cat_tile_name[ind]

        self.N_cat = len(self.z_cat)
        self.log.info('Total number of clusters in catalogue = {}.'.format(self.N_cat))
        self.log.info('SNR cut = {}.'.format(self.qcut))
        self.log.info('Number of clusters above the SNR cut = {}.'.format(self.N_cat))
        self.log.info('The highest redshift = {}'.format(self.z_cat.max()))
        self.log.info('The lowest SNR = {}.'.format(self.q_cat.min()))
        self.log.info('The highest SNR = {}.'.format(self.q_cat.max()))


        self.catalog = pd.DataFrame(
            {
                "z": self.z_cat.byteswap().newbyteorder(),#both.survey.clst_z.byteswap().newbyteorder(),
                "tsz_signal": self.cat_tsz_signal.byteswap().newbyteorder(), #both.survey.clst_y0.byteswap().newbyteorder(),
                "tsz_signal_err": self.cat_tsz_signal_err.byteswap().newbyteorder(),#survey.clst_y0err.byteswap().newbyteorder(),
                "tile_name": self.cat_tile_name.byteswap().newbyteorder()#survey.clst_y0err.byteswap().newbyteorder(),

            }
        )

        # mass bin
        self.lnmmin = np.log(self.binning['M']['Mmin'])
        self.lnmmax = np.log(self.binning['M']['Mmax'])
        self.dlnm = self.binning['M']['dlogM']
        self.lnmarr = np.arange(self.lnmmin+(self.dlnm/2.), self.lnmmax, self.dlnm)
        self.log.info('Number of mass points for theory calculation {}.'.format(len(self.lnmarr)))

        # this is to be consist with szcounts.f90 - maybe switch to linspace?
        self.k = np.logspace(-4, np.log10(4), 200, endpoint=False)
        self.datafile_rms = self.data['rms_file']
        self.datafile_Q = self.data['Q_file']

        if self.selfunc['Qmode'] == 'downsample':
            list = fits.open(os.path.join(self.data_directory, self.datafile_rms))
            file_rms = list[1].data
            self.skyfracs = file_rms['areaDeg2'] * np.deg2rad(1.) ** 2

            filename_Q, ext = os.path.splitext(self.datafile_Q)
            datafile_Q_dwsmpld = os.path.join(self.data_directory,
                                 filename_Q + 'dwsmpld_nbins={}'.format(self.selfunc['dwnsmpl_bins']) + '.npz')

            if os.path.exists(datafile_Q_dwsmpld):
                self.log.info('Reading in binned Q function from file.')
                Qfile = np.load(datafile_Q_dwsmpld)
                self.Q = Qfile['Q_dwsmpld']
                self.tt500 = Qfile['tt500']

            else:
                self.log.info('Reading full Q function.')
                tile_area = np.genfromtxt(os.path.join(self.data_directory, self.data['tile_file']), dtype=str)
                tilename = tile_area[:, 0]
                QFit = nm.signals.QFit(QFitFileName=os.path.join(self.data_directory, self.datafile_Q),
                                       tileNames=tilename, QSource='injection', selFnDir=self.data_directory+'/selFn')
                Nt = len(tilename)
                self.log.info("Initial number of tiles = {}.".format(Nt))

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
                self.skyfracs = self.skyfracs #file_rms['areaDeg2']*np.deg2rad(1.)**2
                self.tname = file_rms['tileName']
                self.log.info("Number of tiles after removing the tiles with zero area = {}. ".format(len(np.unique(self.tname))))
                self.log.info("Number of sky patches = {}.".format(self.skyfracs.size))

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
                        Qdwnsmpld[:, i] = np.average(self.Q[:, test], axis=1, weights=temparea) ##

                self.noise = 0.5*(binned_rms_edges[:-1] + binned_rms_edges[1:])
                self.skyfracs = binned_area
                self.Q = Qdwnsmpld
                self.tiles_dwnsmpld = tiles_dwnsmpld
                # print('len(tiles_dwnsmpld)',tiles_dwnsmpld)
                self.log.info("Number of downsampled sky patches = {}.".format(self.skyfracs.size))

                assert self.noise.shape[0] == self.skyfracs.shape[0] and self.noise.shape[0] == self.Q.shape[1]

                if self.selfunc['save_dwsmpld']:
                    np.savez(datafile_Q_dwsmpld, Q_dwsmpld=Qdwnsmpld, tt500=self.tt500)
                    np.savez(datafile_rms_dwsmpld, noise=self.noise, skyfracs=self.skyfracs)
                    np.save(datafile_tiles_dwsmpld, self.tiles_dwnsmpld)

        elif self.selfunc['Qmode'] == 'full':
            self.log.info('Reading full Q function.')
            tile_area = np.genfromtxt(os.path.join(self.data_directory, self.data['tile_file']), dtype=str)
            tilename = tile_area[:, 0]
            QFit = nm.signals.QFit(QFitFileName=os.path.join(self.data_directory, self.datafile_Q),
                                       tileNames=tilename, QSource='injection', selFnDir=self.data_directory+'/selFn')
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


        if self.selfunc['average_Q']: # currently average_Q should be chosen with Qmode:downsample
            self.Q = np.average(self.Q, axis=1)
            self.noise = np.average(self.noise)
            self.log.info("Number of Q functions = {}.".format(self.Q.ndim))
            self.log.info("Using one averaged Q function for optimisation")
        else:
            self.log.info("Number of Q functions = {}.".format(len(self.Q[0])))


        if self.selfunc['mode'] == 'injection':

            # if hasattr(UnbinnedClusterLikelihood,'qbin') is False:
            #     self.qbins = None

            self.compThetaInterpolator, thetaQ = selfunc.get_completess_inj_theta_y(self.data_directory, self.qcut, self.qbins)
            self.Q = thetaQ

        if self.selfunc['Qmode'] == 'full':
            self.log.info('Reading in full RMS table.')

            list = fits.open(os.path.join(self.data_directory, self.datafile_rms))
            file_rms = list[1].data

            self.noise = file_rms['y0RMS']
            self.skyfracs = file_rms['areaDeg2']*np.deg2rad(1.)**2
            self.tname = file_rms['tileName']
            self.log.info("Number of tiles = {}. ".format(len(np.unique(self.tname))))
            self.log.info("Number of sky patches = {}.".format(self.skyfracs.size))

            tiledict = dict(zip(tilename, np.arange(tile_area[:, 0].shape[0])))
            self.tile_list = [tiledict[key]+1 for key in self.tname]


        self.log.info('Entire survey area = {} deg2.'.format(self.skyfracs.sum()/(np.deg2rad(1.)**2.)))




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

        k = self.k#np.logspace(-4, np.log10(4), 200, endpoint=False)
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

def get_erf_compl(y, qmin, qmax, rms, qcut):

    arg1 = (y/rms - qmax)/np.sqrt(2.)
    if qmin > qcut:
        qlim = qmin
    else:
        qlim = qcut
    arg2 = (y/rms - qlim)/np.sqrt(2.)
    erf_compl = (scipy.special.erf(arg2) - scipy.special.erf(arg1)) / 2.
    return erf_compl

def get_erf(y, rms, cut):
    arg = (y - cut*rms)/np.sqrt(2.)/rms
    erfc = (special.erf(arg) + 1.)/2.
    return erfc


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


def convert_masses(both, marr, zz):

    h = both.theory.get_param("H0") / 100.0
    if both.theorypred['choose_theory'] == 'CCL':
        mf_data = both.theory.get_nc_data()
        md_hmf = mf_data['md']

        if both.theorypred['md_ym'] == '200m':
            md_ym = ccl.halos.MassDef200m(c_m='Bhattacharya13')
        elif both.theorypred['md_ym'] == '200c':
            md_ym = ccl.halos.MassDef200c(c_m='Bhattacharya13')
        elif both.theorypred['md_ym'] == '500c':
            md_ym = ccl.halos.MassDef(500, 'critical')
        else:
            raise NotImplementedError('Only md_hmf = 200m, 200c and 500c currently supported.')

        cosmo = both.theory.get_CCL()['cosmo']
        a = 1. / (1. + zz)
        marr_ymmd = np.array([md_hmf.translate_mass(cosmo, marr / h, ai, md_ym) for ai in a]) * h
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
    marr_500c = np.array([md_hmf.translate_mass(cosmo, marr / h, ai, md_500c) for ai in a]) * h

    return marr_500c

def _splQ(self, theta):
    if self.selfunc['average_Q']:
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

    # if self.name == "Unbinned Clusters":
    #     Ez = Ez.T
    #     DAz = DAz.T

    return ttstar * (mass_500c / MPIVOT_THETA / h) ** alpha_theta * Ez ** (-2. / 3.) * (100. * DAz / 500 / H0) ** (-1.)


# y-m scaling relation for completeness
def _get_y0(self, mass, z, mass_500c, use_Q=True, **params_values_dict):
    # if mass_500c is None:
    #     mass_500c = mass

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
        if self.theorypred['rel_correction']:
            t = -0.008488*(mm*Ez)**(-0.585)
            res = 1.+ 3.79*t - 28.2*(t**2.)
        else:
            res = 1.
        return res

    if use_Q is True:
        theta = _theta(self,mb_500c, z, Ez)
        splQ = _splQ(self,theta)
    else:
        splQ = 1.

    if (self.selfunc['mode'] == 'Qfit' and self.selfunc['average_Q']):
        y0 = A0 * (Ez**2.) * (mb / Mpivot)**(1. + B0) * splQ
        y0 = y0.T
    else:
        # if self.name == "Unbinned Clusters":
        #     Ez = Ez.T
        y0 = A0 * (Ez ** 2.) * (mb / Mpivot) ** (1. + B0) * splQ

    return y0
