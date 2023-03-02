r"""Class to calculate bias models for haloes and galaxies as cobaya Theory classes

"""

import pdb
import numpy as np
from typing import Sequence, Union
from scipy.interpolate import interp1d

from cobaya.theory import Theory
from cobaya.log import LoggedError
from cobaya.theories.cosmo.boltzmannbase import PowerSpectrumInterpolator
try:
    import fastpt as fpt
except:
    "FastPT import failed, FastPT_bias will be unavailable"


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

    def get_Pk_mm_interpolator(self, extrap_kmax=None):

        return self._get_Pk_interpolator(pk_type='mm',
                                         var_pair=self._var_pairs,
                                         nonlinear=self.nonlinear,
                                         extrap_kmax=extrap_kmax)

    def get_Pk_gg_interpolator(self, extrap_kmax=None):

        return self._get_Pk_interpolator(pk_type='gg',
                                         var_pair=self._var_pairs,
                                         nonlinear=self.nonlinear,
                                         extrap_kmax=extrap_kmax)

    def get_Pk_gm_interpolator(self, extrap_kmax=None):

        return self._get_Pk_interpolator(pk_type='gm',
                                         var_pair=self._var_pairs,
                                         nonlinear=self.nonlinear,
                                         extrap_kmax=extrap_kmax)

    def _get_Pk_interpolator(self, pk_type,
                             var_pair=("delta_tot", "delta_tot"), nonlinear=True,
                             extrap_kmax=None):

        nonlinear = bool(nonlinear)

        key = ("Pk_{}_interpolator".format(pk_type), nonlinear, extrap_kmax)\
                + tuple(sorted(var_pair))
        if key in self.current_state:
            return self.current_state[key]

        if pk_type == 'mm':
            pk = self._get_Pk_mm(update_growth=False)
        elif pk_type == 'gm':
            pk = self.get_Pk_gm_grid()
        elif pk_type == 'gg':
            pk = self.get_Pk_gg_grid()

        log_p = True
        sign = 1
        if np.any(pk < 0):
            if np.all(pk < 0):
                sign = -1
            else:
                log_p = False
        if log_p:
            pk = np.log(sign * pk)
        elif (extrap_kmax is not None) and (extrap_kmax > self.k[-1]):
            raise LoggedError(self.log,
                              'Cannot do log extrapolation with zero-crossing pk '
                              'for %s, %s' % var_pair)
        result = PowerSpectrumInterpolator(self.z, self.k, pk, logP=log_p, logsign=sign,
                                           extrap_kmax=extrap_kmax)
        self.current_state[key] = result
        return result


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
            else:
                assert np.allclose(k, self.k) # check we are consistent with ks
            if self.z is None:
                self.z = z
            else:
                assert np.allclose(z, self.z) # check we are consistent with zs

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


class FastPT_bias(Bias):
    # set bias parameters
    params = {'b_11': None, 'b_12': None, 'b_21': None, 'b_22': None,
              # 'b_s1': None, 'b_s2': None, 'b_3nl1': None, 'b_3nl2': None
              }

    def init_fastpt(self, k, C_window = .75, pad_factor = 1):
        self.C_window = C_window
        to_do = ['one_loop_dd', 'dd_bias', 'one_loop_cleft_dd', 'IA_all',
                 'OV', 'kPol', 'RSD', 'IRres']
        n_pad = pad_factor * len(k)
        low_extrap = np.log10(min(k))  # From the example notebook, will change
        high_extrap = np.log10(max(k))  # From the example notebook, will change
        self.fpt_obj = fpt.FASTPT(k, to_do=to_do, low_extrap=low_extrap,
                                  high_extrap=high_extrap, n_pad=n_pad)

    def update_grid(self):
        k, z, pkmm = self.provider.get_Pk_grid(
            var_pair=('delta_tot', 'delta_tot'),
            nonlinear=False)  # needs to be linear

        # result = PowerSpectrumInterpolator(self.z, self.k, pkmm)
        pk_interpolator = []
        for i, zi in enumerate(z):
            pk_interpolator.append(interp1d(k, pkmm[i],
                                            kind='cubic',
                                            fill_value='extrapolate')
                                   )

        if self.k is None:
            self.k = k
        else:
            assert np.allclose(k, self.k) # check we are consistent with ks

        if self.z is None:
            self.z = z
        else:
            assert np.allclose(z, self.z) # check we are consistent with zs

        log10kmin = np.log10(self.k[0])
        log10kmax = np.log10(self.k[-1])
        self.k_fastpt = np.logspace(log10kmin, log10kmax, len(self.k))

        self.pk_fastpt = np.zeros([len(self.z), len(self.k_fastpt)])
        for i, zi in enumerate(self.z):
            self.pk_fastpt[i] = pk_interpolator[i](self.k_fastpt)

    def get_growth(self):
        for pair in self._var_pairs:

            k, z, Pk_mm = self.provider.get_Pk_grid(var_pair=pair,
                                                     nonlinear=False)

            assert(z[0] == 0)

        return np.mean(Pk_mm/Pk_mm[:1], axis = -1)**0.5


    def _get_Pk_gg(self, **params_values_dict):

        b11 = params_values_dict['b_11']
        b12 = params_values_dict['b_12']
        b21 = params_values_dict['b_21']
        b22 = params_values_dict['b_22']
        bs1 = 0.0 # ignore bs terms for now
        bs2 = 0.0 # ignore bs terms for now
        b3nl1 = 0.0 # ignoring these too?
        b3nl2 = 0.0 # ignoring these too?
       # bs1 = params_values_dict['b_s1']
       # bs2 = params_values_dict['b_s2']
       # b3nl1 = params_values_dict['b_3nl1']
       # b3nl2 = params_values_dict['b_3nl1']

        growth = self.get_growth()
        g2 = growth ** 2
        g4 = growth ** 4

        pk_gg = np.zeros([len(self.z), len(self.k)])

        for i, zi in enumerate(self.z):
            P_bias_E = self.fpt_obj.one_loop_dd_bias_b3nl(self.pk_fastpt[i],
                                                          C_window=self.C_window)

            # Output individual terms
            Pd1d1 = g2[i] * self.pk_fastpt[i] + g4[i] * P_bias_E[0]
            Pd1d2 = g4[i] * P_bias_E[2]
            Pd2d2 = g4[i] * P_bias_E[3]
            Pd1s2 = g4[i] * P_bias_E[4]
            Pd2s2 = g4[i] * P_bias_E[5]
            Ps2s2 = g4[i] * P_bias_E[6]
            Pd1p3 = g4[i] * P_bias_E[8]
            s4 = g4[i] * P_bias_E[7]

            pk_gg_fastpt = ((b11*b12) * Pd1d1 +
                            0.5*(b11*b22 + b12*b21) * Pd1d2 +
                            0.25*(b21*b22) * (Pd2d2 - 2.*s4) +
                            0.5*(b11*bs2 + b12*bs1) * Pd1s2 +
                            0.25*(b21*bs2 + b22*bs1) * (Pd2s2 - (4./3.)*s4) +
                            0.25*(bs1*bs2) * (Ps2s2 - (8./9.)*s4) +
                            0.5*(b11 * b3nl2 + b12 * b3nl1) * Pd1p3)

            pk_interpolator = interp1d(self.k_fastpt, pk_gg_fastpt,
                                            kind='cubic',
                                            fill_value='extrapolate')

            pk_gg[i] = pk_interpolator(self.k)

        return pk_gg


    def _get_Pk_gm(self, **params_values_dict):

        b11 = params_values_dict['b_11']
        b21 = params_values_dict['b_21']
        bs1 = 0.0 # ignore bs terms for now
        b3nl1 = 0.0 # ignoring these too?
       # bs1 = params_values_dict['b_s1']
       # b3nl1 = params_values_dict['b_3nl1']

        growth = self.get_growth()
        g2 = growth ** 2
        g4 = growth ** 4

        pk_gm = np.zeros([len(self.z), len(self.k)])

        for i, zi in enumerate(self.z):
            P_bias_E = self.fpt_obj.one_loop_dd_bias_b3nl(self.pk_fastpt[i],
                                                          C_window=self.C_window)

            # Output individual terms
            Pd1d1 = g2[i] * self.pk_fastpt[i] + g4[i] * P_bias_E[0]
            Pd1d2 = g4[i] * P_bias_E[2]
            Pd1s2 = g4[i] * P_bias_E[4]
            Pd1p3 = g4[i] * P_bias_E[8]

            pk_gm_fastpt = (b11 * Pd1d1 +
                            0.5*b21 * Pd1d2 +
                            0.5*bs1 * Pd1s2 +
                            0.5*b3nl1 * Pd1p3)

            pk_interpolator = interp1d(self.k_fastpt, pk_gm_fastpt,
                                            kind='cubic',
                                            fill_value='extrapolate')

            pk_gm[i] = pk_interpolator(self.k)

        return pk_gm


    def calculate(self, state, want_derived=True, **params_values_dict):

        self.update_grid()

        try:
            self.fpt_obj
        except:
            self.init_fastpt(self.k_fastpt)

        state['Pk_gg_grid'] = self._get_Pk_gg(**params_values_dict)
        state['Pk_gm_grid'] = self._get_Pk_gm(**params_values_dict)