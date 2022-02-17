r"""Class to calculate bias models for haloes and galaxies as cobaya Theory classes

"""

import pdb
import numpy as np
from typing import Sequence, Union
from cobaya.theory import Theory
from cobaya.likelihood import Likelihood
import fastpt as fpt
from cobaya.log import LoggedError


class Bias(Theory):

    kmax: float = 10.  # Maximum k (1/Mpc units) for Pk, or zero if not needed
    nonlinear: bool = False  # whether to get non-linear Pk from CAMB/Class
    z: Union[Sequence, np.ndarray] = []  # redshift sampling

    extra_args: dict = {}  # extra (non-parameter) arguments passed to ccl.Cosmology()

    _logz = np.linspace(-3, np.log10(1100), 150)
    _default_z_sampling = 10**_logz
    _default_z_sampling[0] = 0

    def initialize(self):

        self._var_pairs = set()
        # self._var_pairs = [('delta_tot', 'delta_tot')]

    def get_requirements(self):
        return {}

    def must_provide(self, **requirements):

        options = requirements.get('linear_bias') or {}

        self.kmax = max(self.kmax, options.get('kmax', self.kmax))
        self.z = np.unique(np.concatenate(
                            (np.atleast_1d(options.get("z", self._default_z_sampling)),
                            np.atleast_1d(self.z))))

        # Dictionary of the things needed from CAMB/CLASS
        needs = {}

        self.nonlinear = self.nonlinear or options.get('nonlinear', False)
        self._var_pairs.update(
            set((x, y) for x, y in
                options.get('vars_pairs', [('delta_tot', 'delta_tot')])))

        needs['Pk_grid'] = {
                'vars_pairs': self._var_pairs or [('delta_tot', 'delta_tot')],
                'nonlinear': (True, False) if self.nonlinear else False,
                'z': self.z,
                'k_max': self.kmax
            }

        needs['Pk_interpolator'] = {
                'vars_pairs': self._var_pairs or [('delta_tot', 'delta_tot')],
                'nonlinear': (True, False) if self.nonlinear else False,
                'z': self.z,
                'k_max': self.kmax
            }

        assert len(self._var_pairs) < 2, "Bias doesn't support other Pk yet"
        return needs

    def get_growth(self):
        for pair in self._var_pairs:

            k, z, Pk_mm = self.provider.get_Pk_grid(var_pair=pair,
                                                     nonlinear=False)

            assert(z[0] == 0)

        return np.mean(Pk_mm/Pk_mm[:1], axis = -1)**0.5

    def _get_Pk_mm_grid(self):
        for pair in self._var_pairs:

            if self.nonlinear:
                self.k, self.z, Pk_mm = self.provider.get_Pk_grid(var_pair=pair,
                                                            nonlinear=True)
                # Pk_mm = np.flip(Pk_nonlin, axis=0)
            else:
                self.k, self.z, Pk_mm = self.provider.get_Pk_grid(var_pair=pair,
                                                         nonlinear=False)
                # Pk_mm = np.flip(Pk_lin, axis=0)

        return Pk_mm

    def _get_Pk_mm_interpolator(self):

        for pair in self._var_pairs:

            if self.nonlinear:
                Pk_mm = self.provider.get_Pk_interpolator(var_pair=pair, nonlinear=True)
            else:
                Pk_mm = self.provider.get_Pk_interpolator(var_pair=pair, nonlinear=False)

        return Pk_mm

    def get_Pk_gg_grid(self):
        return self._current_state['Pk_gg_grid']

    def get_Pk_gm_grid(self):
        return self._current_state['Pk_gm_grid']

    def get_Pk_gg_interpolator(self):
        return self._current_state['Pk_gg_interpolator']

    def get_Pk_gm_interpolator(self):
        return self._current_state['Pk_gm_interpolator']


class Linear_bias(Bias):

    params = {'b_lin': None}

    def calculate(self, state, want_derived=True, **params_values_dict):

        Pk_mm = self._get_Pk_mm_grid()

        state['Pk_gg_grid'] = params_values_dict['b_lin']**2. * Pk_mm
        state['Pk_gm_grid'] = params_values_dict['b_lin'] * Pk_mm


class FastPT(Bias):

    # set bias parameters
    params = {'b_11': None, 'b_12': None, 'b_21': None, 'b_22': None,
                          'b_s1': None, 'b_s2': None, 'b_3nl1': None, 'b_3nl2': None}

    zs : list = []

    def init_fastpt(self, k, C_window = .75, pad_factor = 1):
        self.C_window = C_window
        to_do = ['one_loop_dd', 'dd_bias', 'one_loop_cleft_dd', 'IA_all',
                 'OV', 'kPol', 'RSD', 'IRres']
        n_pad = pad_factor * len(k)
        low_extrap = np.log10(min(k))  # From the example notebook, will change
        high_extrap = np.log10(max(k))  # From the example notebook, will change
        self.fpt_obj = fpt.FASTPT(k, to_do=to_do, low_extrap=low_extrap,
                                  high_extrap=high_extrap, n_pad=n_pad)

    def calculate(self, state, want_derived=True, **params_values_dict):
        log10kmin = np.log10(1e-5)
        log10kmax = np.log10(self.kmax)
        k = np.logspace(log10kmin, log10kmax, 200)

        Pk_mm = self._get_Pk_mm_interpolator()
        pk = Pk_mm(self.zs, k)

        try:
            self.fpt_obj
        except:
            self.init_fastpt(k)

        pk_gg = np.zeros_like(pk)
        pk_gm = np.zeros_like(pk)

        growth = self.get_growth()
        g2 = growth ** 2
        g4 = growth ** 4

        b11 = params_values_dict['b_11']
        b12 = params_values_dict['b_12']
        b21 = params_values_dict['b_21']
        b22 = params_values_dict['b_22']
        bs1 = params_values_dict['b_s1']
        bs2 = params_values_dict['b_s2']
        b3nl1 = params_values_dict['b_3nl1']
        b3nl2 = params_values_dict['b_3nl1']


        for i, zi in enumerate(self.zs):
            P_bias_E = self.fpt_obj.one_loop_dd_bias_b3nl(pk[i], C_window=self.C_window)

            # Output individual terms
            Pd1d1 = g2[i] * pk[i] + g4[i] * P_bias_E[0]
            Pd1d2 = g4[i] * P_bias_E[2]
            Pd2d2 = g4[i] * P_bias_E[3]
            Pd1s2 = g4[i] * P_bias_E[4]
            Pd2s2 = g4[i] * P_bias_E[5]
            Ps2s2 = g4[i] * P_bias_E[6]
            Pd1p3 = g4[i] * P_bias_E[8]
            s4 = g4[i] * P_bias_E[7]

            pk_gg[i] = ((b11*b12) * Pd1d1 +
                    0.5*(b11*b22 + b12*b21) * Pd1d2 +
                    0.25*(b21*b22) * (Pd2d2 - 2.*s4) +
                    0.5*(b11*bs2 + b12*bs1) * Pd1s2 +
                    0.25*(b21*bs2 + b22*bs1) * (Pd2s2 - (4./3.)*s4) +
                    0.25*(bs1*bs2) * (Ps2s2 - (8./9.)*s4) +
                    0.5*(b11 * b3nl2 + b12 * b3nl1) * Pd1p3)

            pk_gm[i] = (b11 * Pd1d1 +
                    0.5*b21 * Pd1d2 +
                    0.5*bs1 * Pd1s2 +
                    0.5*b3nl1 * Pd1p3)

        state['Pk_gm_grid'] = pk_gm
        state['Pk_gg_grid'] = pk_gg
        state['Pk_mm_grid'] = pk # For consistency (right now it was an interpolator)


class FPTTest(Likelihood):

    def initialize(self):
        pass

    def get_requirements(self):
        return {'Pk_gg_grid': {'b_11': None, 'b_12': None, 'b_21': None, 'b_22': None,
                          'b_s1': None, 'b_s2': None, 'b_3nl1': None, 'b_3nl2': None}}

    def logp(self, **params_values):
        pk_gg = self.provider.get_Pk_gg_grid()

        return np.sum(pk_gg)