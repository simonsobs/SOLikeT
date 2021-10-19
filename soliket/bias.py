r"""Class to calculate bias models for haloes and galaxies as cobaya Theory classes

"""

import numpy as np
from typing import Sequence, Union
from cobaya.theory import Theory

class Bias(Theory):

    def initialize(self):
        kmax: float = 0  # Maximum k (1/Mpc units) for Pk, or zero if not needed
        nonlinear: bool = False  # whether to get non-linear Pk from CAMB/Class
        z: Union[Sequence, np.ndarray] = []  # redshift sampling

    # def get_requirements(self):

    def must_provide(self, **requirements):

        self.kmax = max(self.kmax, options.get('kmax', self.kmax))
        self.z = np.unique(np.concatenate(
                            (np.atleast_1d(options.get("z", self._default_z_sampling)),
                            np.atleast_1d(self.z))))

        # Dictionary of the things needed from CAMB/CLASS
        needs = {}

        needs['Pk_grid'] = {
                'vars_pairs': self._var_pairs or [('delta_tot', 'delta_tot')],
                'nonlinear': (True, False) if self.nonlinear else False,
                'z': self.z,
                'k_max': self.kmax
            }

        return needs

    # def calculate(self, state, want_derived=True, **params_values_dict):

    # def get_Pkgg(self):

    # def get_Pkgm(self):

class Linear_bias(Bias):

    params = {'b_lin' : None}

    def calculate(self, state, want_derived=True, **params_values_dict):

        state['Pk_gg'] = self.b_lin * self.b_lin * Pk_grid
        state['Pk_gm'] = self.b_lin * Pk_grid

    def get_Pkgg(self):
        return self._current_state['Pk_gg']

    def get_Pkgm(self):
        return self._current_state['Pk_gm']
