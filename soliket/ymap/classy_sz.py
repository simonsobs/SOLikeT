from cobaya.theories.classy import classy
# from cobaya.theories.classy import Collector
from copy import deepcopy
from cobaya.theories._cosmo import BoltzmannBase
from cobaya.theory import Theory
from typing import NamedTuple, Sequence, Union, Optional

# class BoltzmannBase_sz(BoltzmannBase):
#     def must_provide(self, **requirements):
#         for k, v in requirements.items():
#             if k == "Cl_sz":
#                 self._must_provide["Cl_sz"] = self._must_provide.get("Cl_sz", {})
#         super().must_provide(**requirements)
#

# Result collector
class Collector(NamedTuple):
    method: str
    args: Sequence = []
    args_names: Sequence = []
    kwargs: dict = {}
    arg_array: Union[int, Sequence] = None
    post: Optional[callable] = None


class classy_sz(classy):
    def must_provide(self, **requirements):
        print('new must provide')
        if requirements.pop("Cl_sz", None):
            print('cl_sz rquired')
            exit(0)
            self._must_provide["Cl_sz"] = self._must_provide.get("Cl_sz", {})
            for k, v in requirements.items():
                if k == "Cl_sz":
                    self.collectors[k] = Collector(
                            method="cl_sz",
                            args_names=[],
                            args=[])
        super().must_provide(**requirements)


    def get_Cl_sz(self):
        cls = {}
        cls = deepcopy(self._current_state["Cl_sz"])
        return cls




    # elif k == "Cl_sz":
    #     self.collectors[k] = Collector(
    #         method="cl_sz",
    #         args_names=[],
    #         args=[])
