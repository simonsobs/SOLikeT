from cobaya.theories.classy import classy
from copy import deepcopy
from typing import NamedTuple, Sequence, Union, Optional
from cobaya.tools import load_module
import logging
import os
import numpy as np
from cobaya.theory import Theory
from pkg_resources import resource_filename
import cosmopower as cp
import scipy
from cobaya.conventions import Const
H_units_conv_factor = {"1/Mpc": 1, "km/s/Mpc": Const.c_km_s}

class cosmopower(classy):
    path_to_trained_models: Optional[str] = resource_filename(
        "cosmopower", "trained_models/CP_paper/CMB/"
    )
    TT_emulator_name: Optional[str] = "Null_TT"
    TE_emulator_name: Optional[str] = "Null_TE"
    EE_emulator_name: Optional[str] = "Null_EE"
    PP_emulator_name: Optional[str] = "Null_PP"
    S8Z_emulator_name: Optional[str] = "Null_S8Z"
    DAZ_emulator_name: Optional[str] = "Null_DAZ"
    HZ_emulator_name: Optional[str] = "Null_HZ"
    DER_emulator_name: Optional[str] = "Null_DERZ"
    PKNL_emulator_name: Optional[str] = "Null_PKNL"
    tt_emu = None
    te_emu = None
    ee_emu = None
    s8z_emu = None
    daz_emu = None
    hz_emu = None
    pp_emu = None
    der_emu = None
    pknl_emu = None
    tt_spectra = None
    te_spectra = None
    ee_spectra = None
    pp_spectra = None

    lensing_lkl = "SOLikeT"

    # pkslice = 10
    # pp_spectra = None

    def initialize(self):
        """Importing CLASS from the correct path, if given, and if not, globally."""
        self.classy_module = self.is_installed()
        if not self.classy_module:
            raise NotInstalledError(
                self.log, "Could not find class. Check error message above.")
        from classy_sz import Class, CosmoSevereError, CosmoComputationError
        global CosmoComputationError, CosmoSevereError
        self.classy = Class()
        super(classy,self).initialize()
        # Add general CLASS stuff
        self.extra_args["output"] = self.extra_args.get("output", "")
        if "sBBN file" in self.extra_args:
            self.extra_args["sBBN file"] = (
                self.extra_args["sBBN file"].format(classy=self.path))
        # Derived parameters that may not have been requested, but will be necessary later
        self.derived_extra = []
        # print(self.likelihood)
        # exit(0)
        if 'Null' not in self.TT_emulator_name:
            self.tt_emu = cp.cosmopower_NN(restore=True,
                                      restore_filename=self.path_to_trained_models+self.TT_emulator_name)
        if 'Null' not in self.TE_emulator_name:
            self.te_emu = cp.cosmopower_PCAplusNN(restore=True,
                                             restore_filename=self.path_to_trained_models+self.TE_emulator_name)
        if 'Null' not in self.EE_emulator_name:
            self.ee_emu = cp.cosmopower_NN(restore=True,
                                  restore_filename=self.path_to_trained_models+self.EE_emulator_name)
        if 'Null' not in self.S8Z_emulator_name:
            self.s8z_emu = cp.cosmopower_NN(restore=True,
                                  restore_filename=self.path_to_trained_models+self.S8Z_emulator_name)
        if 'Null' not in self.DAZ_emulator_name:
            self.daz_emu = cp.cosmopower_NN(restore=True,
                                  restore_filename=self.path_to_trained_models+self.DAZ_emulator_name)
        if 'Null' not in self.HZ_emulator_name:
            self.hz_emu = cp.cosmopower_NN(restore=True,
                                  restore_filename=self.path_to_trained_models+self.HZ_emulator_name)
        if 'Null' not in self.PP_emulator_name:
            self.pp_emu = cp.cosmopower_NN(restore=True,
                                  restore_filename=self.path_to_trained_models+self.PP_emulator_name)
        # print(self.DER_emulator_name)
        if 'Null' not in self.DER_emulator_name:
            # print(self.DER_emulator_name)
            # exit(0)
            self.der_emu = cp.cosmopower_NN(restore=True,
                                      restore_filename=self.path_to_trained_models+self.DER_emulator_name)
        if 'Null' not in self.PKNL_emulator_name:
            self.pknl_emu= cp.cosmopower_NN(restore=True,
                                  restore_filename=self.path_to_trained_models+self.PKNL_emulator_name)



        self.log.info("Initialized!")
        # self.log.info("using " + self.PP_emulator_name )
        # self.log.info("using " + self.PP_emulator_name + " and " + 'DER_cp_NN_Oct22B')

        # exit(0)

    # # here modify if you want to bypass stuff in the class computation
    def calculate(self, state, want_derived=True, **params_values_dict):
        # print("[calculate] Bypassing class computation")
        # Set parameters
        params_values = params_values_dict.copy()
        params_values['ln10^{10}A_s'] = params_values.pop("logA")

        params_dict = {}
        for k,v in zip(params_values.keys(),params_values.values()):
            params_dict[k]=[v]
        # print('\n')
        # print('new params:')
        # print(params_dict)
        # exit(0)
        if self.tt_emu is not None:
            self.tt_spectra = self.tt_emu.ten_to_predictions_np(params_dict)
        if self.te_emu is not None:
            self.te_spectra = self.te_emu.predictions_np(params_dict)
        if self.ee_emu is not None:
            self.ee_spectra = self.ee_emu.ten_to_predictions_np(params_dict)

        params_dict_pp = params_dict.copy()
        params_dict_pp.pop('tau_reio')

        if self.pp_emu is not None:
            self.pp_spectra = self.pp_emu.ten_to_predictions_np(params_dict)
        # print('pp spetrum ',self.pp_spectra)
        # params_dict['z_pk_save_nonclass'] = [1.]
        if self.der_emu is not None:
            self.der_params = self.der_emu.ten_to_predictions_np(params_dict)

        if self.hz_emu is not None:
            self.hz  = self.hz_emu.ten_to_predictions_np(params_dict_pp)

            self.hz_interp = scipy.interpolate.interp1d(
                                            np.linspace(0.,20.,5000),
                                            self.hz[0],
                                            kind='linear',
                                            axis=-1,
                                            copy=True,
                                            bounds_error=None,
                                            fill_value=np.nan,
                                            assume_sorted=False)
        if self.daz_emu is not None:
            self.daz  = self.daz_emu.predictions_np(params_dict_pp)

            self.daz_interp = scipy.interpolate.interp1d(
                                            np.linspace(0.,20.,5000),
                                            self.daz[0],
                                            kind='linear',
                                            axis=-1,
                                            copy=True,
                                            bounds_error=None,
                                            fill_value=np.nan,
                                            assume_sorted=False)

        if self.s8z_emu is not None:
            self.s8z  = self.s8z_emu.predictions_np(params_dict_pp)
            # print(self.s8z)
            self.s8z_interp = scipy.interpolate.interp1d(
                                            np.linspace(0.,20.,5000),
                                            self.s8z[0],
                                            kind='linear',
                                            axis=-1,
                                            copy=True,
                                            bounds_error=None,
                                            fill_value=np.nan,
                                            assume_sorted=False)

        for product, collector in self.collectors.items():
            arg_array = self.collectors[product].arg_array

            if product == 'fsigma8':
                # print('dealing with fsigma8')

                method = getattr(self, collector.method)
                # print(method)
                arg_array = self.collectors[product].arg_array
                if isinstance(arg_array, int):
                    arg_array = np.atleast_1d(arg_array)
                if arg_array is None:
                    print('dealing with arg_array none case... calling method')
                elif isinstance(arg_array, Sequence) or isinstance(arg_array, np.ndarray):
                    arg_array = np.array(arg_array)
                    if len(arg_array.shape) == 1:
                        # print('dealing with arg_array shape 1 case... calling method')
                        # if more than one vectorised arg, assume all vectorised in parallel
                        n_values = len(self.collectors[product].args[arg_array[0]])
                        state[product] = np.zeros(n_values)
                        args = deepcopy(list(self.collectors[product].args))
                        args.append(params_values_dict)
                        for i in range(n_values):
                            for arg_arr_index in arg_array:
                                args[arg_arr_index] = \
                                    self.collectors[product].args[arg_arr_index][i]
                            state[product][i] = method(*args)
                            # print('fs8: ',state[product][i])
                else:
                    raise LoggedError(self.log, "Variable over which to do an array call "
                                                f"not known: arg_array={arg_array}")
            if ('Pk_grid' in product) or ('comoving_radial_distance') in product:
                method = getattr(self, collector.method)
                args = deepcopy(list(self.collectors[product].args))
                args.append(params_values_dict)
                state[product] = method(*args)
                if collector.post:
                    state[product] = collector.post(*state[product])


    # get the required new observable
    def get_Cl(self,ell_factor=True,units="FIRASmuK2"):

        cls = {}
        cls['ell'] = np.arange(20000)
        # print(cls['ell'])
        cls['tt'] = np.zeros(20000)
        cls['te'] = np.zeros(20000)
        cls['ee'] = np.zeros(20000)
        cls['pp'] = np.zeros(20000)
        if self.tt_spectra is not None:
            nl = len(self.tt_spectra[0])
            # print('nl:',nl)
            cls['tt'][2:nl+2] = (2.7255e6)**2.*self.tt_spectra[0].copy()
            if ell_factor==False:
                lcp = np.asarray(cls['ell'][2:nl+2])
                cls['tt'][2:nl+2] *= 1./(lcp*(lcp+1.)/2./np.pi)

        if self.te_spectra is not None:
            cls['te'][2:nl+2] = (2.7255e6)**2.*self.te_spectra[0].copy()
            if ell_factor==False:
                lcp = np.asarray(cls['ell'][2:nl+2])
                cls['te'][2:nl+2] *= 1./(lcp*(lcp+1.)/2./np.pi)
        if self.ee_spectra is not None:
            cls['ee'][2:nl+2] = (2.7255e6)**2.*self.ee_spectra[0].copy()
            if ell_factor==False:
                lcp = np.asarray(cls['ell'][2:nl+2])
                cls['ee'][2:nl+2] *= 1./(lcp*(lcp+1.)/2./np.pi)
        if self.pp_spectra is not None:
            nl = len(self.pp_spectra[0])
            if self.lensing_lkl ==  "SOLikeT":
                cls['pp'][2:nl+2] = self.pp_spectra[0].copy()/4. ## this is clkk... works for so lensinglite lkl
            else:
                # here for the planck lensing lkl, using lfactor option gives:
                lcp = np.asarray(cls['ell'][2:nl+2])
                cls['pp'][2:nl+2] = self.pp_spectra[0].copy()/(lcp*(lcp+1.))**2.
                cls['pp'][2:nl+2] *= (lcp*(lcp+1.))**2./2./np.pi



        return cls

    def get_param(self, p):
        translated = self.translate_param(p)
        if translated == 'rs_drag':
            return self.rs_drag()

        if p == 'omegam':
            return self.Omega_m()

    @classmethod
    def is_installed(cls, **kwargs):
        return load_module('cosmopower')


    #################################
    # gives an estimation of f(z)*sigma8(z) at the scale of 8 h/Mpc, computed as (d sigma8/d ln a)
    def effective_f_sigma8(self, z, z_step=0.1,params_values_dict={}):
        """
        effective_f_sigma8(z)

        Returns the time derivative of sigma8(z) computed as (d sigma8/d ln a)

        Parameters
        ----------
        z : float
                Desired redshift
        z_step : float
                Default step used for the numerical two-sided derivative. For z < z_step the step is reduced progressively down to z_step/10 while sticking to a double-sided derivative. For z< z_step/10 a single-sided derivative is used instead.

        Returns
        -------
        (d ln sigma8/d ln a)(z) (dimensionless)
        """

        s8z_interp =  self.s8z_interp
        # we need d sigma8/d ln a = - (d sigma8/dz)*(1+z)

        # if possible, use two-sided derivative with default value of z_step
        if z >= z_step:
            result = (s8z_interp(z-z_step)-s8z_interp(z+z_step))/(2.*z_step)*(1+z)
            # return (s8z_interp(z-z_step)-s8z_interp(z+z_step))/(2.*z_step)*(1+z)
        else:
            # if z is between z_step/10 and z_step, reduce z_step to z, and then stick to two-sided derivative
            if (z > z_step/10.):
                z_step = z
                result = (s8z_interp(z-z_step)-s8z_interp(z+z_step))/(2.*z_step)*(1+z)
                # return (s8z_interp(z-z_step)-s8z_interp(z+z_step))/(2.*z_step)*(1+z)
            # if z is between 0 and z_step/10, use single-sided derivative with z_step/10
            else:
                z_step /=10
                result = (s8z_interp(z)-s8z_interp(z+z_step))/z_step*(1+z)
                # return (s8z_interp(z)-s8z_interp(z+z_step))/z_step*(1+z)
        # print('fsigma8 result : ',result)
        return result

    def rs_drag(self):
        return self.der_params[0][13]

    def Omega_m(self):
        params_values = self.current_state['params'].copy()
        result = (params_values['omega_b']+params_values['omega_cdm'])*(100./params_values['H0'])**2.
        return result

    def get_angular_diameter_distance(self, z):
        return np.array(self.daz_interp(z))


    def get_Hubble(self, z,units="km/s/Mpc"):
        return np.array(self.hz_interp(z)*H_units_conv_factor[units])


    def get_pk_and_k_and_z(self,params_values_dict={}):
        nz = 15 # number of z-points in redshift data [21oct22] --> set to 80
        zmax = 4. # max redshift of redshift data [21oct22] --> set to 4 because boltzmannbase.py wants to extrapolate
        z_arr = np.linspace(0.,zmax,nz) # z-array of redshift data [21oct22] oct 26 22: nz = 1000, zmax = 20

        nk = 5000
        ndspl = 10
        k_arr = np.geomspace(1e-4,50.,nk)[::ndspl]  # oct 26 22 : (1e-4,50.,5000), jan 10: ndspl
        # k_arr = np.geomspace(1e-4,50.,nk) # test

        # scaling factor for the pk emulator:
        ls = np.arange(2,5000+2)[::ndspl] # jan 10 ndspl
        # ls = np.arange(2,5000+2) # test

        dls = ls*(ls+1.)/2./np.pi
        # params_values = self.current_state['params'].copy()
        params_values = params_values_dict.copy()
        params_values['ln10^{10}A_s'] = params_values.pop("logA")


        # # naive implementation... this is way slower:
        # params_dict = {}
        # for k,v in zip(params_values.keys(),params_values.values()):
        #     params_dict[k]=np.repeat(v, nz)
        # params_dict_pp = params_dict.copy()
        # params_dict_pp.pop('tau_reio')
        # params_dict_pp['z_pk_save_nonclass'] = z_arr
        # predicted_pknl_spectrum = self.pknl_emu.predictions_np(params_dict_pp)
        # print(np.shape(predicted_pknl_spectrum))

        params_dict = {}
        for k,v in zip(params_values.keys(),params_values.values()):
            params_dict[k]=[v]
        predicted_pknl_spectrum_z = []
        # def get_pk_cp(zz):
        #     params_dict_pp = params_dict.copy()
        #     params_dict_pp.pop('tau_reio')
        #     params_dict_pp['z_pk_save_nonclass'] = [zz]
        #     return  self.pknl_emu.predictions_np(params_dict_pp)[0]
        for zp in z_arr:
            params_dict_pp = params_dict.copy()
            params_dict_pp.pop('tau_reio')
            params_dict_pp['z_pk_save_nonclass'] = [zp]
            predicted_pknl_spectrum_z.append(self.pknl_emu.predictions_np(params_dict_pp)[0])
            # if zp>4.:
            #     predicted_pknl_spectrum_z.append(0.*k_arr)
            # predicted_pknl_spectrum_z.append(get_pk_cp(zp))
        # get_pk_cp = np.vectorize(get_pk_cp)
        # predicted_pknl_spectrum_z = get_pk_cp([0.1,0.3])
        # print(np.shape(predicted_pknl_spectrum_z))
        # exit(0)
        predicted_pknl_spectrum = np.asarray(predicted_pknl_spectrum_z)

        # print(np.shape(predicted_pknl_spectrum))
        # print(predicted_pknl_spectrum[0][:10])
        # exit(0)


        pknl = 10.**predicted_pknl_spectrum
        pknl_re =  ((dls)**-1*pknl)
        pknl_re = np.transpose(pknl_re)


        # return pknl_re[0:k_arr.size:self.pkslice][:], k_arr[0:k_arr.size:self.pkslice], z_arr
        return pknl_re, k_arr, z_arr


    def z_of_r (self,z_array,params_values_dict={}):
        return np.array(self.daz_interp(z_array))*(1.+z_array), np.array(self.hz_interp(z_array))


# this just need to be there as it's used to fill-in self.collectors in must_provide:
class Collector(NamedTuple):
    method: str
    args: Sequence = []
    args_names: Sequence = []
    kwargs: dict = {}
    arg_array: Union[int, Sequence] = None
    post: Optional[callable] = None
