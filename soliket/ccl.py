"""
.. module:: soliket.ccl

:Synopsis: A simple CCL wrapper for Cobaya.
:Author: Pablo Lemos.

.. |br| raw:: html

   <br />

The `Core Cosmology Library (CCL) <https://ccl.readthedocs.io/en/latest/>`_ is a
standardized library of routines to calculate basic observables used in cosmology.
It will be the standard analysis package used by the LSST
Dark Energy Science Collaboration (DESC).

This Theory is a simple CCL wrapper with function to return CCL cosmo object, and
(optional) result of calling various custom methods on the ccl object.
The idea is this is included with the CCL package, so it can easily be used as a Cobaya
component whenever CCL is installed, here for now.

First version by AL. Untested example of usage at
https://github.com/cmbant/SZCl_like/blob/methods/szcl_like/szcl_like.py

``get_CCL`` results a dictionary of results, where ``results['cosmo']`` is the
CCL cosmology object.

Classes that need other CCL-computed results (without additional free parameters), should
pass them in the requirements list.

e.g. a Likelihood with :func:`~soliket.ccl.CCL.get_requirements` returning
``{'CCL': {'methods:{'name': self.method}}}`` [where self is the Theory instance]
will have ``results['name']`` set to the result of ``self.method(cosmo)`` being
called with the CCL cosmo object.

The Likelihood class can therefore handle for itself which results specifically it needs
from CCL, and just give the method to return them (to be called and cached by Cobaya with
the right parameters at the appropriate time).

Alternatively the Likelihood can compute what it needs from ``results['cosmo']``,
however in this case it will be up to the Likelihood to cache the results
appropriately itself.

Note that this approach preclude sharing results other than the cosmo object itself
between different likelihoods.

Also note lots of things still cannot be done consistently in CCL, so this is far from
general.

.. note::

   **If you use this cosmological code, please cite it as:**
   |br|
   N. Chiasari et al.
   *Core Cosmology Library: Precision Cosmological Predictions for LSST*
   (`arXiv:1812.05995 <https://arxiv.org/abs/1812.05995>`_)
   (`Homepage <https://github.com/LSSTDESC/CCL>`_)
   |br|
   CCL is open source and available for free under the *BSD-3-Clause license*.

Usage
-----

To use CCL, simply add ``soliket.CCL`` as a theory code to your run settings. The
likelihood then needs to have ``CCL`` as a requirement, optionally with any of the
following additional reqs:

* ``Pk_grid``
* ``Hubble``
* ``comoving_radial_distance``
* ``fsigma8``
* ``sigma8_z``

Then, to obtain the results, evaluate the contents of ``self.provider.get_CCL()`` in
the likelihood.
"""

# For Cobaya docs see
# https://cobaya.readthedocs.io/en/devel/theory.html
# https://cobaya.readthedocs.io/en/devel/theories_and_dependencies.html

import numpy as np
from typing import Sequence
from cobaya.theory import Theory
from cobaya.tools import LoggedError


class CCL(Theory):
    """A theory code wrapper for CCL."""
    _logz = np.linspace(-3, np.log10(1100), 150)
    _default_z_sampling = 10 ** _logz
    _default_z_sampling[0] = 0
    kmax: float
    z: np.ndarray
    nonlinear: bool

    def initialize(self) -> None:
        try:
            import pyccl as ccl
        except ImportError:
            raise LoggedError(self.log, "Could not import ccl. Install pyccl to use ccl.")
        else:
            self.ccl = ccl

        self._var_pairs = set()
        self._required_results = {}

    def get_requirements(self) -> set:
        # These are currently required to construct a CCL cosmology object.
        # Ultimately CCL should depend only on observable not parameters
        return {'omch2', 'ombh2'}

    def must_provide(self, **requirements) -> dict:
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

    def get_can_support_params(self) -> Sequence[str]:
        # return any nuisance parameters that CCL can support
        return []

    def calculate(self, state: dict, want_derived: bool = True,
                  **params_values_dict) -> bool:
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
        sigma8 =   self.provider.get_param('sigma8')
        n_s = self.provider.get_param('ns')
        mnu = self.provider.get_param('mnu')
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
                    cosmo = self.ccl.CosmologyCalculator(
                        Omega_c=Omega_c,
                        Omega_b=Omega_b,
                        h=h,
                        sigma8=sigma8,
                        n_s=n_s,
                        m_nu=mnu,
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
                    cosmo = self.ccl.CosmologyCalculator(
                        Omega_c=Omega_c,
                        Omega_b=Omega_b,
                        h=h,
                        sigma8=sigma8,
                        n_s=n_s,
                        m_nu=mnu,
                        background={'a': a,
                                    'chi': distance,
                                    'h_over_h0': E_of_z},
                        pk_linear={'a': a,
                                   'k': k,
                                   'delta_matter:delta_matter': Pk_lin} # noqa E501
                        )

        state['CCL'] = {'cosmo': cosmo 'ccl': self.ccl}

        for required_result, method in self._required_results.items():
            state['CCL'][required_result] = method(cosmo)

    def get_CCL(self):
        return self._current_state['CCL']
