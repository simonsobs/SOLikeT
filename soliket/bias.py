r"""Class to calculate bias models for haloes and galaxies as cobaya Theory classes

"""

import pdb
import numpy as np
from typing import Sequence, Union
from scipy.interpolate import interp1d

from cobaya.theory import Theory
from cobaya.log import LoggedError
try:
    from velocileptors.EPT.cleft_kexpanded_resummed_fftw import RKECLEFT
except:
    "velocileptors import failed, LPT_bias will be unavailable"


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
        self.k = None
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


    def _get_Pk_mm(self, update_growth=True):

        for pair in self._var_pairs:

            if self.nonlinear:
                k, z, Pk_mm = self.provider.get_Pk_grid(var_pair=pair,
                                                        nonlinear=True)
            else:
                k, z, Pk_mm = self.provider.get_Pk_grid(var_pair=pair,
                                                        nonlinear=False)
            if self.k is None:
                self.k = k
            if self.z is None:
                self.z = z

            if update_growth:
                assert(z[0] == 0)
                self.Dz = np.mean(Pk_mm / Pk_mm[:1], axis=-1)**0.5

        return Pk_mm

    def get_Pk_gg_grid(self):
        return self._current_state['Pk_gg_grid']

    def get_Pk_gm_grid(self):
        return self._current_state['Pk_gm_grid']


class Linear_bias(Bias):

    params = {'b_lin': None}

    def calculate(self, state, want_derived=True, **params_values_dict):

        Pk_mm = self._get_Pk_mm(update_growth=False) # growth not needed for linear bias

        state['Pk_gg_grid'] = params_values_dict['b_lin']**2. * Pk_mm
        state['Pk_gm_grid'] = params_values_dict['b_lin'] * Pk_mm


class LPT_bias(Bias):

    params = {'b1g1': None,
              'b2g1': None,
              # 'bsg1': None,
              'b1g2': None,
              'b2g2': None,
              # 'bsg2': None
              }

    def init_cleft(self, k_filter=None):

        self.update_lpt_table()

        if k_filter is not None:
            self.wk_low = 1 - np.exp(-(self.k / k_filter)**2)
        else:
            self.wk_low = np.ones_like(self.k)


    def update_lpt_table(self):

        k, z, Pk_mm = self.provider.get_Pk_grid(var_pair=('delta_tot', 'delta_tot'),
                                                nonlinear=False)

        if self.k is None:
            self.k = k
        else:
            assert np.allclose(k, self.k) # check we are consistent with ks

        if self.z is None:
            self.z = z
        else:
            assert np.allclose(z, self.z) # check we are consistent with zs

        self.cleft_obj = RKECLEFT(self.k, Pk_mm[0],
                                  extrap_min=np.floor(np.log10(self.k[0])),
                                  extrap_max=np.ceil(np.log10(self.k[-1])))

        self.lpt_table = []

        assert(z[0] == 0)
        self.Dz = np.mean(Pk_mm / Pk_mm[:1], axis=-1)**0.5

        for D in self.Dz:
            self.cleft_obj.make_ptable(D=D,
                                       kmin=self.k[0],
                                       kmax=self.k[-1],
                                       nk=self.k.size)
            self.lpt_table.append(self.cleft_obj.pktable)

        self.lpt_table = np.array(self.lpt_table)


    def _get_Pk_gg(self, **params_values_dict):

        b11 = params_values_dict['b1g1']
        b21 = params_values_dict['b2g1']
        # bs1 = params_values_dict['bsg1']
        bs1 = 0.0 # ignore bs terms for now
        b12 = params_values_dict['b1g2']
        b22 = params_values_dict['b2g2']
        # bs2 = params_values_dict['bsg2']
        bs2 = 0.0 # ignore bs terms for now

        bL11 = b11 - 1
        bL12 = b12 - 1

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

        pgg_interpolated = interp1d(cleft_k[0], pgg,
                                    kind='cubic',
                                    fill_value='extrapolate')(self.k)

        return pgg_interpolated

    def _get_Pk_gm(self, **params_values_dict):

        b1 = params_values_dict['b1g1']
        b2 = params_values_dict['b2g1']
        # bs = params_values_dict['bsg1']
        bs = 0.0 # ignore bs terms for now

        bL1 = b1 - 1

        Pdmdm = self.lpt_table[:, :, 1]
        Pdmd1 = 0.5 * self.lpt_table[:, :, 2]
        pgm = Pdmdm + bL1 * Pdmd1

        Pdmd2 = 0.5 * self.lpt_table[:, :, 4]
        Pdms2 = 0.25 * self.lpt_table[:, :, 7]

        pgm += (b2 * Pdmd2 +
                bs * Pdms2)

        cleft_k = self.lpt_table[:, :, 0]

        pgm_interpolated = interp1d(cleft_k[0], pgm,
                                    kind='cubic',
                                    fill_value='extrapolate')(self.k)

        return pgm_interpolated

    def calculate(self, state, want_derived=True, **params_values_dict):

        try:
            self.cleft_obj
        except:
            self.init_cleft(k_filter=1.e-2)

        self.update_lpt_table()

        state['Pk_gg_grid'] = self._get_Pk_gg(**params_values_dict)
        state['Pk_gm_grid'] = self._get_Pk_gm(**params_values_dict)
