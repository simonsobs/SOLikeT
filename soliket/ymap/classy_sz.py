from cobaya.theories.classy import classy
from copy import deepcopy
from typing import NamedTuple, Sequence, Union, Optional


class classy_sz(classy):

    def must_provide(self, **requirements):
        if "Cl_sz" in requirements:
            # make sure cobaya still runs as it does for standard classy
            requirements.pop("Cl_sz")
            # specify the method to collect the new observable
            self.collectors["Cl_sz"] = Collector(
                    method="cl_sz", # name of the method in classy.pyx
                    args_names=[],
                    args=[])
        super().must_provide(**requirements)

    # get the required new observable
    def get_Cl_sz(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_sz"])
        return cls

    # @classmethod
    # def is_installed(cls, **kwargs):
    #     try:
    #         return load_module(
    #             'classy_sz', path=classy_build_path, min_version=cls._classy_repo_version)
    #     except ImportError:
    #         if path is not None and path.lower() != "global":
    #             log.error("Couldn't find the CLASS python interface at '%s'. "
    #                       "Are you sure it has been installed there?", path)
    #         else:
    #             log.error("Could not import global CLASS installation. "
    #                       "Specify a Cobaya or CLASS installation path, "
    #                       "or install the CLASS Python interface globally with "
    #                       "'cd /path/to/class/python/ ; python setup.py install'")
    #         return False
# this just need to be there as it's used to fill-in self.collectors in must_provide:
class Collector(NamedTuple):
    method: str
    args: Sequence = []
    args_names: Sequence = []
    kwargs: dict = {}
    arg_array: Union[int, Sequence] = None
    post: Optional[callable] = None
