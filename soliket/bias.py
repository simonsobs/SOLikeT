r"""Class to calculate bias models for haloes and galaxies as cobaya Theory classes

"""

import pdb
import numpy as np
from typing import Sequence, Union
from scipy.interpolate import interp1d

from cobaya.theory import Theory
from cobaya.log import LoggedError
from velocileptors.EPT.cleft_kexpanded_resummed_fftw import RKECLEFT


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

        assert len(self._var_pairs) < 2, "Bias doesn't support other Pk yet"
        return needs

    def _get_growth(self):
        for pair in self._var_pairs:

            k, z, Pk_mm = self.provider.get_Pk_grid(var_pair=pair,
                                                     nonlinear=False)

            assert(z[0] == 0)

        return np.mean(Pk_mm / Pk_mm[:1], axis=-1)**0.5

    def _get_Pk_mm(self):

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

    def get_Pk_gg_grid(self):
        return self._current_state['Pk_gg_grid']

    def get_Pk_gm_grid(self):
        return self._current_state['Pk_gm_grid']


class Linear_bias(Bias):

    params = {'b_lin': None}

    def calculate(self, state, want_derived=True, **params_values_dict):

        Pk_mm = self._get_Pk_mm()

        state['Pk_gg_grid'] = params_values_dict['b_lin']**2. * Pk_mm
        state['Pk_gm_grid'] = params_values_dict['b_lin'] * Pk_mm


class LPT_bias(Bias):

    params = {'b11': None,
              'b21': None,
              'bs1': None,
              'b12': None,
              'b22': None,
              'bs2': None}

    def init_cleft(self, k_filter=None):

        Pk_mm = self._get_Pk_mm()

        if k_filter is not None:
            self.wk_low = 1-np.exp(-(self.k/k_filter)**2)
        else:
            self.wk_low = np.ones_like(self.k)
        
        self.cleft_obj = RKECLEFT(self.k, Pk_mm[0],
                                  extrap_min=np.floor(np.log10(self.k[0])),
                                  extrap_max=np.ceil(np.log10(self.k[-1])))

        self.lpt_table = []

        for D in self._get_growth():
            self.cleft_obj.make_ptable(D=D, kmin=self.k[0], kmax=self.k[-1], nk=self.k.size)
            self.lpt_table.append(self.cleft_obj.pktable)

        self.lpt_table = np.array(self.lpt_table)

    def _get_Pk_gg(self, Pk_mm, **params_values_dict):

        b11 = params_values_dict['b11']# * np.ones_like(Pk_mm)
        b21 = params_values_dict['b21']# * np.ones_like(Pk_mm)
        bs1 = params_values_dict['bs1']# * np.ones_like(Pk_mm)
        b12 = params_values_dict['b12']# * np.ones_like(Pk_mm)
        b22 = params_values_dict['b22']# * np.ones_like(Pk_mm)
        bs2 = params_values_dict['bs2']# * np.ones_like(Pk_mm)

        bL11 = b11 - 1
        bL12 = b12 - 1

        if self.nonlinear:
            pgg = (b11 * b12) * Pk_mm
        else:
            Pdmdm = self.lpt_table[:, :, 1]
            Pdmd1 = 0.5 * self.lpt_table[:, :, 2]
            Pd1d1 = self.lpt_table[:, :, 3]
            pgg = (Pdmdm + (bL11 + bL12) * Pdmd1 + (bL11 * bL12) * Pd1d1)

        Pdmd2 = 0.5 * self.lpt_table[:, :, 4]
        Pd1d2 = 0.5 * self.lpt_table[:, :, 5]
        Pd2d2 = self.lpt_table[:, :, 6] * self.wk_low[None, :]
        Pdms2 = 0.25 * self.lpt_table[:, :, 7]
        Pd1s2 = 0.25 * self.lpt_table[:, :, 8]
        Pd2s2 = 0.25 * self.lpt_table[:, :, 9] * self.wk_low[None, :]
        Ps2s2 = 0.25 * self.lpt_table[:, :, 10] * self.wk_low[None, :]

        pgg += ((b21 + b22) * Pdmd2 +
                (bs1 + bs2) * Pdms2 +
                (bL11 * b22 + bL12 * b21) * Pd1d2 +
                (bL11 * bs2 + bL12 * bs1) * Pd1s2 +
                (b21 * b22) * Pd2d2 +
                (b21 * bs2 + b22 * bs1) * Pd2s2 +
                (bs1 * bs2) * Ps2s2)

        cleft_k = self.lpt_table[:, :, 0]

        # pdb.set_trace()

        # spectra = {r'$(1,1)$':self.lpt_table[:,:,1],r'$(1,b_1)$':0.5*self.lpt_table[:,:,2], r'$(b_1,b_1)$': self.lpt_table[:,:,3],r'$(1,b_2)$':0.5*self.lpt_table[:,:,4], r'$(b_1,b_2)$': 0.5*self.lpt_table[:,:,5],  r'$(b_2,b_2)$': self.lpt_table[:,:,6],r'$(1,b_s)$':0.5*self.lpt_table[:,:,7], r'$(b_1,b_s)$': 0.5*self.lpt_table[:,:,8],  r'$(b_2,b_s)$':0.5*self.lpt_table[:,:,9], r'$(b_s,b_s)$':self.lpt_table[:,:,10],r'$(1,b_3)$':0.5*self.lpt_table[:,:,11],r'$(b_1,b_3)$': 0.5*self.lpt_table[:,:,12]}

        pgg_interpolated = interp1d(cleft_k[0], pgg, kind='cubic', fill_value='extrapolate')(self.k)

        return pgg_interpolated

    def calculate(self, state, want_derived=True, **params_values_dict):

        try:
            self.cleft_obj
        except:
            self.init_cleft(k_filter=1.e-1)

        Pk_mm = self._get_Pk_mm()

        state['Pk_gg_grid'] = self._get_Pk_gg(Pk_mm, **params_values_dict)
