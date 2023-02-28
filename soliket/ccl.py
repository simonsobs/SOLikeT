"""
.. module:: soliket.ccl

:Synopsis: A simple CCL wrapper for Cobaya.

Simple CCL wrapper with function to return CCL cosmo object, and (optional) result of
calling various custom methods on the ccl object. The idea is this is included with the
CCL package, so it can easily be used as a Cobaya component whenever CCL is installed,
here for now.

First version by AL. Untested example of usage at
https://github.com/cmbant/SZCl_like/blob/methods/szcl_like/szcl_like.py

get_CCL results a dictionary of results, where results['cosmo'] is the CCL cosmology
object.

Classes that need other CCL-computed results (without additional free parameters), should
pass them in the requirements list.

e.g. a Likelihood with get_requirements() returning
{'CCL': {'methods:{'name': self.method}}} [where self is the Theory instance] will have
results['name'] set to the result of self.method(cosmo) being called with theCCL cosmo
object.

The Likelihood class can therefore handle for itself which results specifically it needs
from CCL, and just give the method to return them (to be called and cached by Cobaya with
the right parameters at the appropriate time).

Alternatively the Likelihood can compute what it needs from results['cosmo'], however in
this case it will be up to the Likelihood to cache the results appropriately itself.

Note that this approach preclude sharing results other than the cosmo object itself
between different likelihoods.

Also note lots of things still cannot be done consistently in CCL, so this is far from
general.

April 2021:
-----------
Second version by PL. Using CCL's newly implemented cosmology calculator.
"""

# For Cobaya docs see
# https://cobaya.readthedocs.io/en/devel/theory.html
# https://cobaya.readthedocs.io/en/devel/theories_and_dependencies.html

import numpy as np
from typing import Sequence, Union
from cobaya.theory import Theory
import pyccl as ccl


class CCL(Theory):
    """
    
    """
    # Options for Pk.
    # Default options can be set globally, and updated from requirements as needed
    kmax: float = 0  # Maximum k (1/Mpc units) for Pk, or zero if not needed
    nonlinear: bool = False  # whether to get non-linear Pk from CAMB/Class
    z: Union[Sequence, np.ndarray] = []  # redshift sampling
    extra_args: dict = {}  # extra (non-parameter) arguments passed to ccl.Cosmology()

    _logz = np.linspace(-3, np.log10(1100), 150)
    _default_z_sampling = 10**_logz
    _default_z_sampling[0] = 0

    def initialize(self):
        self._var_pairs = set()
        self._required_results = {}

    def get_requirements(self):
        # These are currently required to construct a CCL cosmology object.
        # Ultimately CCL should depend only on observable not parameters
        return {'omch2', 'ombh2'}

    def must_provide(self, **requirements):
        # requirements is dictionary of things requested by likelihoods
        # Note this may be called more than once

        if 'CCL' not in requirements:
            return {}
        options = requirements.get('CCL') or {}
        if 'methods' in options:
            self._required_results.update(options['methods'])

        self.kmax = max(self.kmax, options.get('kmax', self.kmax))
        self.z = np.unique(np.concatenate(
                            (np.atleast_1d(options.get("z", self._default_z_sampling)),
                            np.atleast_1d(self.z))))

        # Dictionary of the things CCL needs from CAMB/CLASS
        needs = {}

        if self.kmax:
            self.nonlinear = self.nonlinear or options.get('nonlinear', False)
            # CCL currently only supports ('delta_tot', 'delta_tot'), but call allow
            # general as placeholder
            self._var_pairs.update(
                set((x, y) for x, y in
                    options.get('vars_pairs', [('delta_tot', 'delta_tot')])))

            needs['Pk_grid'] = {
                'vars_pairs': self._var_pairs or [('delta_tot', 'delta_tot')],
                'nonlinear': (True, False) if self.nonlinear else False,
                'z': self.z,
                'k_max': self.kmax
            }

        needs['Hubble'] = {'z': self.z}
        needs['comoving_radial_distance'] = {'z': self.z}

        needs['fsigma8'] = {'z': self.z}
        needs['sigma8_z'] = {'z': self.z}

        assert len(self._var_pairs) < 2, "CCL doesn't support other Pk yet"
        return needs

    def get_can_support_params(self):
        # return any nuisance parameters that CCL can support
        return []

    def calculate(self, state, want_derived=True, **params_values_dict):
        # calculate the general CCL cosmo object which likelihoods can then use to get
        # what they need (likelihoods should cache results appropriately)
        # get our requirements from self.provider

        distance = self.provider.get_comoving_radial_distance(self.z)
        hubble_z = self.provider.get_Hubble(self.z)
        H0 = hubble_z[0]
        h = H0 / 100
        E_of_z = hubble_z / H0

        Omega_c = self.provider.get_param('omch2') / h ** 2
        Omega_b = self.provider.get_param('ombh2') / h ** 2
        # Array z is sorted in ascending order. CCL requires an ascending scale factor
        # as input
        # Flip the arrays to make them a function of the increasing scale factor.
        # If redshift sampling is changed, check that it is monotonically increasing
        distance = np.flip(distance)
        E_of_z = np.flip(E_of_z)

        # Array z is sorted in ascending order. CCL requires an ascending scale
        # factor as input
        a = 1. / (1 + self.z[::-1])
        # growth = ccl.background.growth_factor(cosmo, a)
        # fgrowth = ccl.background.growth_rate(cosmo, a)

        if self.kmax:
            for pair in self._var_pairs:
                # Get the matter power spectrum:
                k, z, Pk_lin = self.provider.get_Pk_grid(var_pair=pair, nonlinear=False)
                Pk_lin = np.flip(Pk_lin, axis=0)

                if self.nonlinear:
                    _, z, Pk_nonlin = self.provider.get_Pk_grid(var_pair=pair,
                                                                nonlinear=True)
                    Pk_nonlin = np.flip(Pk_nonlin, axis=0)

                    # Create a CCL cosmology object. Because we are giving it background
                    # quantities, it should not depend on the cosmology parameters given
                    cosmo = ccl.CosmologyCalculator(
                                                    Omega_c=Omega_c,
                                                    Omega_b=Omega_b,
                                                    h=h,
                                                    sigma8=0.8,
                                                    n_s=0.96,
                                                    background={'a': a,
                                                                'chi': distance,
                                                                'h_over_h0': E_of_z},
                                                    pk_linear={'a': a,
                                                               'k': k,
                                                               'delta_matter:delta_matter': Pk_lin}, # noqa E501
                                                    pk_nonlin={'a': a,
                                                               'k': k,
                                                               'delta_matter:delta_matter': Pk_nonlin} # noqa E501
                                                    )

                else:
                    cosmo = ccl.CosmologyCalculator(
                                                    Omega_c=Omega_c,
                                                    Omega_b=Omega_b,
                                                    h=h,
                                                    sigma8=0.8,
                                                    n_s=0.96,
                                                    background={'a': a,
                                                                'chi': distance,
                                                                'h_over_h0': E_of_z},
                                                    pk_linear={'a': a,
                                                               'k': k,
                                                               'delta_matter:delta_matter': Pk_lin} # noqa E501
                                                    )

        state['CCL'] = {'cosmo': cosmo}
        for required_result, method in self._required_results.items():
            state['CCL'][required_result] = method(cosmo)

    def get_CCL(self):
        return self._current_state['CCL']
