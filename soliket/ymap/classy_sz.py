from cobaya.theories.classy import classy
from copy import deepcopy
from cobaya.theories._cosmo import BoltzmannBase
from cobaya.theory import Theory

# class BoltzmannBase_sz(BoltzmannBase):
#     def must_provide(self, **requirements):
#         for k, v in requirements.items():
#             if k == "Cl_sz":
#                 self._must_provide["Cl_sz"] = self._must_provide.get("Cl_sz", {})
#         super().must_provide(**requirements)
#

class classy_sz(classy,BoltzmannBase):
    def must_provide(self, **requirements):
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
