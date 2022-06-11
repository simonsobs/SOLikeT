"""
Simple CCL theory wrapper that returns the cosmology object
and optionally a number of methods depending only on that
object.

This is based on an earlier implementation by Antony Lewis:
https://github.com/cmbant/SZCl_like/blob/methods/szcl_like/ccl.py

`get_CCL` results a dictionary of results, where `results['cosmo']`
is the CCL cosmology object.

Classes that need other CCL-computed results (without additional
free parameters), should pass them in the requirements list.

e.g. a `Likelihood` with `get_requirements()` returning
`{'CCL': {'methods:{'name': self.method}}}`
[where self is the Likelihood instance] will have
`results['name']` set to the result
of `self.method(cosmo)` being called with the CCL cosmo
object.

The `Likelihood` class can therefore handle for itself which
results specifically it needs from CCL, and just give the
method to return them (to be called and cached by Cobaya with
the right parameters at the appropriate time).

Alternatively the `Likelihood` can compute what it needs from
`results['cosmo']`, however in this case it will be up to the
`Likelihood` to cache the results appropriately itself.

Note that this approach precludes sharing results other than
the cosmo object itself between different likelihoods.

Also note lots of things still cannot be done consistently
in CCL, so this is far from general.
"""

import numpy as np
import pyccl as ccl
from typing import NamedTuple, Sequence, Union, Optional, Callable
from copy import deepcopy

from cobaya.theory import Theory
from cobaya.log import LoggedError
from cobaya.tools import Pool1D, Pool2D, PoolND, combine_1d

# Result collector
# NB: cannot use kwargs for the args, because the CLASS Python interface
#     is C-based, so args without default values are not named.
class Collector(NamedTuple):
    method: str
    args: Sequence = []
    args_names: Sequence = []
    kwargs: dict = {}
    arg_array: Union[int, Sequence, None] = None
    z_pool: Optional[PoolND] = None
    post: Optional[Callable] = None

class CCL(Theory):
    """
    This implements CCL as a `Theory` object that takes in
    cosmological parameters directly (i.e. cannot be used
    downstream from camb/CLASS.
    """
    # CCL options
    transfer_function: str = 'boltzmann_camb'
    matter_pk: str = 'halofit'
    baryons_pk: str = 'nobaryons'
    md_hmf: str = '200m'
    # Params it can accept
    params = {'Omega_c': None,
              'Omega_b': None,
              'h': None,
              'n_s': None,
              'sigma8': None,
              'm_nu': None}

    def initialize(self):
        self.collectors = {}
        self._required_results = {}

    def get_requirements(self):
        return {}

    def must_provide(self, **requirements):
        # requirements is dictionary of things requested by likelihoods
        # Note this may be called more than once

        # CCL currently has no way to infer the required inputs from
        # the required outputs
        # So a lot of this is fixed
        # if 'CCL' not in requirements:
        #     return {}
        # options = requirements.get('CCL') or {}
        # if 'methods' in options:
        #     self._required_results.update(options['methods'])

        self._required_results.update(requirements)

        for k, v in self._required_results.items():

            if k == "Hubble":
                self.set_collector_with_z_pool(
                    k, v["z"], "Hubble", args_names=["z"], arg_array=0)

            elif k == "angular_diameter_distance":
                self.set_collector_with_z_pool(
                    k, v["z"], "angular_diameter_distance", args_names=["z"], arg_array=0)

        return {}

    def get_can_provide_params(self):
        # return any derived quantities that CCL can compute
        return ['H0']

    def get_param(self, p: str) -> float:
        """
        Interface function for likelihoods and other theory components to get derived
        parameters.
        """
        return self.current_state["derived"][p]

    def get_can_support_params(self):
        # return any nuisance parameters that CCL can support
        return []

    def calculate(self, state, want_derived=True, **params_values_dict):
        # Generate the CCL cosmology object which can then be used downstream
        cosmo = ccl.Cosmology(Omega_c=self.provider.get_param('Omega_c'),
                              Omega_b=self.provider.get_param('Omega_b'),
                              h=self.provider.get_param('h'),
                              n_s=self.provider.get_param('n_s'),
                              sigma8=self.provider.get_param('sigma8'),
                              T_CMB=2.7255,
                              m_nu=self.provider.get_param('m_nu'),
                              transfer_function=self.transfer_function,
                              matter_power_spectrum=self.matter_pk,
                              baryons_power_spectrum=self.baryons_pk)


        # Compute sigma8 (we should actually only do this if required -- TODO)
        state['derived'] = {'H0': cosmo.cosmo.params.H0}
        for req_res, attrs in self._required_results.items():
            if req_res == 'Hubble':
                a = 1./(1. + attrs['z'])
                state[req_res] = ccl.h_over_h0(cosmo, a)*cosmo.cosmo.params.H0
            elif req_res == 'angular_diameter_distance':
                a = 1./(1. + attrs['z'])
                state[req_res] = ccl.angular_diameter_distance(cosmo, a)
            elif req_res == 'Pk_interpolator':
                state[req_res] = None
            elif req_res == 'nc_data':
                if self.md_hmf == '200m':
                    md = ccl.halos.MassDef200m(c_m='Bhattacharya13')
                elif self.md_hmf == '200c':
                    md = ccl.halos.MassDef200c(c_m='Bhattacharya13')
                elif self.md_hmf == '500c':
                    md = ccl.halos.MassDef(500, 'critical', c_m_relation='Bhattacharya13')
                else:
                    raise NotImplementedError('Only md_hmf = 200m, 200c and 500c currently supported.')
                mf = ccl.halos.MassFuncTinker08(cosmo, mass_def=md)
                state[req_res] = {'HMF': mf,
                                  'md': md}
            elif req_res == 'CCL':
                state[req_res] = {'cosmo': cosmo}
            elif attrs is None:
                pass
                # General derived parameters
                # if req_res not in self.derived_extra:
                #     self.derived_extra += [req_res]

    def set_collector_with_z_pool(self, k, zs, method, args=(), args_names=(),
                                  kwargs=None, arg_array=None, post=None, d=1):
        """
        Creates a collector for a z-dependent quantity, keeping track of the pool of z's.
        If ``z`` is an arg, i.e. it is in ``args_names``, then omit it in the ``args``,
        e.g. ``args_names=["a", "z", "b"]`` should be passed together with
        ``args=[a_value, b_value]``.
        """
        if k in self.collectors:
            z_pool = self.collectors[k].z_pool
            z_pool.update(zs)
        else:
            Pool = {1: Pool1D, 2: Pool2D}[d]
            z_pool = Pool(zs)
        # Insert z as arg or kwarg
        kwargs = kwargs or {}
        if d == 1 and "z" in kwargs:
            kwargs = deepcopy(kwargs)
            kwargs["z"] = z_pool.values
        elif d == 1 and "z" in args_names:
            args = deepcopy(args)
            i_z = args_names.index("z")
            args = list(args[:i_z]) + [z_pool.values] + list(args[i_z:])
        elif d == 2 and "z1" in args_names and "z2" in args_names:
            # z1 assumed appearing before z2!
            args = deepcopy(args)
            i_z1 = args_names.index("z1")
            i_z2 = args_names.index("z2")
            args = (list(args[:i_z1]) + [z_pool.values[:, 0]] + list(args[i_z1:i_z2]) +
                    [z_pool.values[:, 1]] + list(args[i_z2:]))
        else:
            raise LoggedError(
                self.log,
                f"I do not know how to insert the redshift for collector method {method} "
                f"of requisite {k}")
        self.collectors[k] = Collector(
            method=method, z_pool=z_pool, args=args, args_names=args_names, kwargs=kwargs,
            arg_array=arg_array, post=post)

    def get_CCL(self):
        """
        Get dictionary of CCL computed quantities.
        results['cosmo'] contains the initialized CCL Cosmology object.
        Other entries are computed by methods passed in as the requirements

        :return: dict of results
        """
        return self._current_state['CCL']

    def get_nc_data(self):
        """
        Get dictionary of CCL computed quantities.
        results['cosmo'] contains the initialized CCL Cosmology object.
        Other entries are computed by methods passed in as the requirements

        :return: dict of results
        """
        return self._current_state['nc_data']

    def _get_z_dependent(self, quantity, z, pool=None):
        if pool is None:
            pool = self.collectors[quantity].z_pool
        try:
            i_kwarg_z = pool.find_indices(z)
        except ValueError:
            raise LoggedError(self.log, f"{quantity} not computed for all z requested. "
                                        f"Requested z are {z}, but computed ones are "
                                        f"{pool.values}.")
        return np.array(self.current_state[quantity], copy=True)[i_kwarg_z]

    def get_Hubble(self, z):
        r"""
        Returns the Hubble rate at the given redshift(s) ``z``.
        The redshifts must be a subset of those requested when
        :func:`~BoltzmannBase.must_provide` was called.
        The available units are ``"km/s/Mpc"`` (i.e. :math:`cH(\mathrm(Mpc)^{-1})`) and
        ``1/Mpc``.
        """

        return self._get_z_dependent("Hubble", z)

    def get_angular_diameter_distance(self, z):
        r"""
        Returns the physical angular diameter distance in :math:`\mathrm{Mpc}` to the
        given redshift(s) ``z``.
        The redshifts must be a subset of those requested when
        :func:`~BoltzmannBase.must_provide` was called.
        """
        return self._get_z_dependent("angular_diameter_distance", z)

    def get_Pk_interpolator(self, var_pair=("delta_tot", "delta_tot"), nonlinear=True,
                            extrap_kmin=None, extrap_kmax=None):

        return None