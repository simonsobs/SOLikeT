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


class cosmopower(classy):
    path_to_trained_models: Optional[str] = resource_filename(
        "cosmopower", "trained_models/CP_paper/CMB/"
    )
    TT_emulator_name: Optional[str] = "TT"
    TE_emulator_name: Optional[str] = "TE"
    EE_emulator_name: Optional[str] = "EE"
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
        self.tt_emu = cp.cosmopower_NN(restore=True,
                                  restore_filename=self.path_to_trained_models+self.TT_emulator_name)
        self.te_emu = cp.cosmopower_PCAplusNN(restore=True,
                                         restore_filename=self.path_to_trained_models+self.TE_emulator_name)
        self.ee_emu = cp.cosmopower_NN(restore=True,
                                  restore_filename=self.path_to_trained_models+self.EE_emulator_name)
        self.log.info("Initialized!")

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

        self.tt_spectra = self.tt_emu.ten_to_predictions_np(params_dict)
        self.te_spectra = self.te_emu.predictions_np(params_dict)
        self.ee_spectra = self.ee_emu.ten_to_predictions_np(params_dict)
        # print('new spectra:')
        # print(self.tt_spectra)
        # print('\n')
        # print(self.te_spectra)
        # print(self.ee_spectra)
        # exit(0)

    # get the required new observable
    def get_Cl(self,ell_factor=True):

        cls = {}
        cls['tt'] = np.zeros(20000)
        cls['te'] = np.zeros(20000)
        cls['ee'] = np.zeros(20000)

        nl = len(self.tt_spectra[0])
        cls['tt'][2:nl+2] = (2.7255e6)**2.*self.tt_spectra[0].copy()
        cls['te'][2:nl+2] = (2.7255e6)**2.*self.te_spectra[0].copy()
        cls['ee'][2:nl+2] = (2.7255e6)**2.*self.ee_spectra[0].copy()
        # print('dls:',cls)
        return cls


    @classmethod
    def is_installed(cls, **kwargs):
        return load_module('cosmopower')




# this just need to be there as it's used to fill-in self.collectors in must_provide:
class Collector(NamedTuple):
    method: str
    args: Sequence = []
    args_names: Sequence = []
    kwargs: dict = {}
    arg_array: Union[int, Sequence] = None
    post: Optional[callable] = None
