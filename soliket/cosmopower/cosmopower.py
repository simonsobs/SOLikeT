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
    tt_emu = None
    te_emu = None
    ee_emu = None
    s8z_emu = None
    daz_emu = None
    hz_emu = None
    pp_emu = None
    der_emu = None
    tt_spectra = None
    te_spectra = None
    ee_spectra = None
    pp_spectra = None

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


        # self.s8z  = self.s8z_emu.predictions_np(params_dict_pp)
        # print(params_dict)
        # print(self.pp_spectra)
        # print(self.der_params)
        # exit(0)

        # if np.sum(self.pp_spectra[0])<1e-10 or np.sum(self.pp_spectra[0])>1:
        #     print(params_dict)
        #     print('sum: %.5e'%np.sum(self.pp_spectra[0]))
        #     print('h:',self.der_params[0][0])
        #     print('sigma8:',self.der_params[0][1])
        #     omega_m = (params_dict['omega_b'][0]+params_dict['omega_cdm'][0])/self.der_params[0][0]**2.
        #     print('omega_m:',omega_m)
        #     print('check!\n')

        # print('new spectra:')
        # print(self.tt_spectra)
        # print('\n')
        # print(self.te_spectra)
        # print(self.ee_spectra)
        # exit(0)
        # derived_requested=False
        # requested = [self.translate_param(p) for p in (
        #     self.output_params if derived_requested else [])]
        # requested_and_extra = dict.fromkeys(set(requested).union(self.derived_extra))
        # print('requested params: ',requested)
        # print('requested_and_extra params: ',requested_and_extra)
        for product, collector in self.collectors.items():
            # print(product,collector)
            if "sigma8" in self.collectors:
                self.collectors["sigma8"].args[0] = 8 / self.classy.h()
        if product == 'fsigma8':
            # print('dealing with fsigma8')

            method = getattr(self, collector.method)
            # print(method)
            arg_array = self.collectors[product].arg_array
            # print(arg_array)
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
                    # print('n_values ',n_values)
                    # print('arg ',args)
                    args.append(params_values_dict)
                    # print('arg ',args)
                    # exit()

                    for i in range(n_values):
                        for arg_arr_index in arg_array:
                            args[arg_arr_index] = \
                                self.collectors[product].args[arg_arr_index][i]
                        state[product][i] = method(*args)
                        # print('fs8: ',state[product][i])
            else:
                raise LoggedError(self.log, "Variable over which to do an array call "
                                            f"not known: arg_array={arg_array}")
        #
        # if product == 'Hubble':
        #     print('dealing with Hubble')
        #
        #     method = getattr(self, collector.method)
        #     print(method)
        #     arg_array = self.collectors[product].arg_array
        #     print(arg_array)
        #     if isinstance(arg_array, int):
        #         arg_array = np.atleast_1d(arg_array)
        #     if arg_array is None:
        #         print('dealing with arg_array none case... calling method')
        #     elif isinstance(arg_array, Sequence) or isinstance(arg_array, np.ndarray):
        #         arg_array = np.array(arg_array)
        #         if len(arg_array.shape) == 1:
        #             print('dealing with arg_array shape 1 case... calling method')
        #             # if more than one vectorised arg, assume all vectorised in parallel
        #             n_values = len(self.collectors[product].args[arg_array[0]])
        #             state[product] = np.zeros(n_values)
        #             args = deepcopy(list(self.collectors[product].args))
        #             print('n_values ',n_values)
        #             print('arg ',args)
        #             args.append(params_values_dict)
        #             print('arg ',args)
        #             # exit()
        #
        #             for i in range(n_values):
        #                 for arg_arr_index in arg_array:
        #                     args[arg_arr_index] = \
        #                         self.collectors[product].args[arg_arr_index][i]
        #                 state[product][i] = method(*args)
        #                 print('hz: ',state[product][i])
        #     else:
        #         raise LoggedError(self.log, "Variable over which to do an array call "
        #                                     f"not known: arg_array={arg_array}")
        #
        #     # print('\n -----')
        # # exit(0)
        # if product == 'angular_diameter_distance':
        #     print('dealing with angular_diameter_distance')
        #
        #     method = getattr(self, collector.method)
        #     print(method)
        #     arg_array = self.collectors[product].arg_array
        #     print(arg_array)
        #     if isinstance(arg_array, int):
        #         arg_array = np.atleast_1d(arg_array)
        #     if arg_array is None:
        #         print('dealing with arg_array none case... calling method')
        #     elif isinstance(arg_array, Sequence) or isinstance(arg_array, np.ndarray):
        #         arg_array = np.array(arg_array)
        #         if len(arg_array.shape) == 1:
        #             print('dealing with arg_array shape 1 case... calling method')
        #             # if more than one vectorised arg, assume all vectorised in parallel
        #             n_values = len(self.collectors[product].args[arg_array[0]])
        #             state[product] = np.zeros(n_values)
        #             args = deepcopy(list(self.collectors[product].args))
        #             print('n_values ',n_values)
        #             print('arg ',args)
        #             args.append(params_values_dict)
        #             print('arg ',args)
        #             # exit()
        #
        #             for i in range(n_values):
        #                 for arg_arr_index in arg_array:
        #                     args[arg_arr_index] = \
        #                         self.collectors[product].args[arg_arr_index][i]
        #                 state[product][i] = method(*args)
        #                 print('da: ',state[product][i])
        #     else:
        #         raise LoggedError(self.log, "Variable over which to do an array call "
        #                                     f"not known: arg_array={arg_array}")

            # print('\n -----')
        # exit(0)

    # get the required new observable
    def get_Cl(self,ell_factor=True):

        cls = {}
        cls['tt'] = np.zeros(20000)
        cls['te'] = np.zeros(20000)
        cls['ee'] = np.zeros(20000)
        cls['pp'] = np.zeros(20000)
        if self.tt_spectra is not None:
            nl = len(self.tt_spectra[0])
            cls['tt'][2:nl+2] = (2.7255e6)**2.*self.tt_spectra[0].copy()
        if self.te_spectra is not None:
            cls['te'][2:nl+2] = (2.7255e6)**2.*self.te_spectra[0].copy()
        if self.ee_spectra is not None:
            cls['ee'][2:nl+2] = (2.7255e6)**2.*self.ee_spectra[0].copy()

        # cls['pp'][2:nl+2] = (2.7255e6)**2.*self.pp_spectra[0].copy()/2./np.pi
        if self.pp_spectra is not None:
            nl = len(self.pp_spectra[0])
            cls['pp'][2:nl+2] = self.pp_spectra[0].copy()/4.
        # print(cls)
        # exit(0)
        # print('dls:',cls)
        return cls

    def get_param(self, p):
        translated = self.translate_param(p)
        # print('current_state',self.current_state)
        # print('translated param vvv',translated)
        # exit(0)
        # for pool in ["params", "derived"]:
        #     value = (self.current_state[pool] or {}).get(translated, None)
        #     if value is not None:
        #         print('value:',value)
        #         return value
        if translated == 'rs_drag':
            # print('v:',self.rs_drag())
            return self.rs_drag()

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
        params_values = params_values_dict.copy()
        params_values['ln10^{10}A_s'] = params_values.pop("logA")

        params_dict = {}
        for k,v in zip(params_values.keys(),params_values.values()):
            params_dict[k]=[v]
        params_dict_pp = params_dict.copy()
        params_dict_pp.pop('tau_reio')
        # print(params_dict_pp)

        self.s8z  = self.s8z_emu.predictions_np(params_dict_pp)
        # print(self.s8z)
        s8z_interp = scipy.interpolate.interp1d(
                                        np.linspace(0.,20.,5000),
                                        self.s8z[0],
                                        kind='linear',
                                        axis=-1,
                                        copy=True,
                                        bounds_error=None,
                                        fill_value=np.nan,
                                        assume_sorted=False)

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
        # exit()
    def rs_drag(self):
        # self.compute(["thermodynamics"])
        # print('rsdrag:',self.der_params[0][13])
        return self.der_params[0][13]

    def get_angular_diameter_distance(self, z):
        """
        angular_distance(z)

        Return the angular diameter distance (exactly, the quantity defined by Class
        as index_bg_ang_distance in the background module)

        Parameters
        ----------
        z : float
                Desired redshift
        """
        # print('computing angular diameter distance')
        # print('self.current_state:',self.current_state)
        params_values = self.current_state['params'].copy()
        params_values['ln10^{10}A_s'] = params_values.pop("logA")
        # print(params_values)
        # exit(0)
        params_dict = {}
        for k,v in zip(params_values.keys(),params_values.values()):
            params_dict[k]=[v]
        params_dict_pp = params_dict.copy()
        params_dict_pp.pop('tau_reio')
        # print(params_dict_pp)
        self.daz  = self.daz_emu.predictions_np(params_dict_pp)

        daz_interp = scipy.interpolate.interp1d(
                                        np.linspace(0.,20.,5000),
                                        self.daz[0],
                                        kind='linear',
                                        axis=-1,
                                        copy=True,
                                        bounds_error=None,
                                        fill_value=np.nan,
                                        assume_sorted=False)
        # print('daz interp:',daz_interp(z))
        # exit(0)
        return np.array(daz_interp(z))


    def get_Hubble(self, z,units="km/s/Mpc"):
        # print('computing Hubble distance')
        # print('self.current_state:',self.current_state)
        params_values = self.current_state['params'].copy()
        params_values['ln10^{10}A_s'] = params_values.pop("logA")
        # print(params_values)
        # exit(0)
        params_dict = {}
        for k,v in zip(params_values.keys(),params_values.values()):
            params_dict[k]=[v]
        params_dict_pp = params_dict.copy()
        params_dict_pp.pop('tau_reio')
        # print(params_dict_pp)
        self.hz  = self.hz_emu.ten_to_predictions_np(params_dict_pp)

        hz_interp = scipy.interpolate.interp1d(
                                        np.linspace(0.,20.,5000),
                                        self.hz[0],
                                        kind='linear',
                                        axis=-1,
                                        copy=True,
                                        bounds_error=None,
                                        fill_value=np.nan,
                                        assume_sorted=False)
        # print('hz interp:',hz_interp(z)*H_units_conv_factor[units])
        # exit(0)


        # units =
        return np.array(hz_interp(z)*H_units_conv_factor[units])

# this just need to be there as it's used to fill-in self.collectors in must_provide:
class Collector(NamedTuple):
    method: str
    args: Sequence = []
    args_names: Sequence = []
    kwargs: dict = {}
    arg_array: Union[int, Sequence] = None
    post: Optional[callable] = None
