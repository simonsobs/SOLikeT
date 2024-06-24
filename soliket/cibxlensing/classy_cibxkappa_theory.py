from cobaya.theories.classy import classy
from copy import deepcopy
from typing import NamedTuple, Sequence, Union, Optional
from cobaya.tools import load_module
import logging
import os

class classy_cibxkappa(classy):
    def initialize(self):
        """Importing CLASS from the correct path, if given, and if not, globally."""
        self.classy_module = self.is_installed()
        if not self.classy_module:
            raise NotInstalledError(
                self.log, "Could not find CLASS_SZ. Check error message above.")
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
        self.log.info("Initialized!")


        # # class_sz default params for lkl
        # self.extra_args["output"] = 'tSZ_1h'
        # self.extra_args["multipoles_sz"] = 'P15'
        # self.extra_args['nlSZ'] = 18


    # # here modify if you want to bypass stuff in the class computation
    # def calculate(self, state, want_derived=True, **params_values_dict):
    #     print("Bypassing class_sz")

    # def calculate(self, state, want_derived=True, **params_values_dict):
    #     #Adjust class_sz CIB Parameter Names
    #     if 'Redshift_evolution_of_dust_temperature' in params_values_dict:
    #        params_values_dict['Redshift evolution of dust temperature'] = params_values_dict.pop('Redshift_evolution_of_dust_temperature')
    #     print('\nRenamed parameter successfully\n')
    #     import ipdb; ipdb.set_trace()
    #     super().calculate(state, want_derived= want_derived, **params_values_dict)
    

    def must_provide(self, **requirements):
        if "Cl_kappa_cib" in requirements:
            # make sure cobaya still runs as it does for standard classy
            requirements.pop("Cl_kappa_cib")
            # specify the method to collect the new observable
            self.collectors["Cl_kappa_cib"] = Collector(
                    method="cl_lens_cib", # name of the method in classy.pyx
                    args_names=[],
                    args=[])

        super().must_provide(**requirements)

    # get the required new observable
    def get_Cl_kappa_cib(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_kappa_cib"])
        return cls

    @classmethod
    def is_installed(cls, **kwargs):
        return load_module('classy_sz')


# this just need to be there as it's used to fill-in self.collectors in must_provide:
class Collector(NamedTuple):
    method: str
    args: Sequence = []
    args_names: Sequence = []
    kwargs: dict = {}
    arg_array: Union[int, Sequence] = None
    post: Optional[callable] = None
