"""

Simple CCL wrapper with function to return CCL cosmo object, and (optional) result of
calling various custom methods on the ccl object. The idea is this is included with the CCL 
package, so it can easily be used as a Cobaya component whenever CCL is installed, here for now.

First version by AL. Untested example of usage at
https://github.com/cmbant/SZCl_like/blob/methods/szcl_like/szcl_like.py

get_CCL results a dictionary of results, where results['cosmo'] is the CCL cosmology object.

Classes that need other CCL-computed results (without additional free parameters), should
pass them in the requirements list.

e.g. a Likelihood with get_requirements() returning {'CCL': {'methods:{'name': self.method}}}
[where self is the Theory instance] will have results['name'] set to the result
of self.method(cosmo) being called with the CCL cosmo object.

The Likelihood class can therefore handle for itself which results specifically it needs from CCL,
and just give the method to return them (to be called and cached by Cobaya with the right
parameters at the appropriate time).

Alternatively the Likelihood can compute what it needs from results['cosmo'], however in this
case it will be up to the Likelihood to cache the results appropriately itself.

Note that this approach preclude sharing results other than the cosmo object itself between different likelihoods.

Also note lots of things still cannot be done consistently in CCL, so this is far from general.

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
from .gaussian import GaussianData, GaussianLikelihood
import pyccl as ccl


class CCL(Theory):
    # Options for Pk.
    # Default options can be set globally, and updated from requirements as needed
    kmax: float = 0  # Maximum k (1/Mpc units) for Pk, or zero if not needed
    nonlinear: bool = False  # whether to get non-linear Pk from CAMB/Class
    z: Union[Sequence, np.ndarray] = []  # redshift sampling
    extra_args: dict = {}  # extra (non-parameter) arguments passed to ccl.Cosmology()

    #_default_z_sampling = np.linspace(0, 10, 150)
    _logz = np.linspace(-3, np.log10(1100), 150)
    _default_z_sampling = 10**_logz
    _default_z_sampling[0] = 0
    #_default_z_sampling[-1] = 1100

    def initialize(self):
        self._var_pairs = set()
        self._required_results = {}

    def get_requirements(self):
        # These are currently required to construct a CCL cosmology object.
        # Ultimately CCL should depend only on observable not parameters
        # 'As' could be substituted by sigma8.
        return {'omch2', 'ombh2'}

    def must_provide(self, **requirements):
        # requirements is dictionary of things requested by likelihoods
        # Note this may be called more than once

        # CCL currently has no way to infer the required inputs from the required outputs
        # So a lot of this is fixed
        if 'CCL' not in requirements:
            return {}
        options = requirements.get('CCL') or {}
        if 'methods' in options:
            self._required_results.update(options['methods'])

        self.kmax = max(self.kmax, options.get('kmax', self.kmax))
        self.z = np.unique(np.concatenate(
            (np.atleast_1d(options.get("z", self._default_z_sampling)), np.atleast_1d(self.z))))

        # Dictionary of the things CCL needs from CAMB/CLASS
        needs = {}

        if self.kmax:
            self.nonlinear = self.nonlinear or options.get('nonlinear', False)
            # CCL currently only supports ('delta_tot', 'delta_tot'), but call allow
            # general as placeholder
            self._var_pairs.update(
                set((x, y) for x, y in
                    options.get('vars_pairs', [('delta_nonu', 'delta_nonu')])))

            needs['Pk_grid'] = {
                'vars_pairs': self._var_pairs or [('delta_nonu', 'delta_nonu')],
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
        h = H0/100
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
        #growth = ccl.background.growth_factor(cosmo, a)
        #fgrowth = ccl.background.growth_rate(cosmo, a)


        if self.kmax:
            for pair in self._var_pairs:
                # Get the matter power spectrum:
                k, z, Pk_lin = self.provider.get_Pk_grid(var_pair=pair, nonlinear=False)
                Pk_lin = np.flip(Pk_lin, axis=0)

                if self.nonlinear:
                    _, z, Pk_nonlin = self.provider.get_Pk_grid(var_pair=pair, nonlinear=True)
                    Pk_nonlin = np.flip(Pk_nonlin, axis=0)

                    # Create a CCL cosmology object. Because we are giving it background 
                    # quantities, it should not depend on the cosmology parameters given
                    cosmo = ccl.CosmologyCalculator(
                        Omega_c=Omega_c, Omega_b=Omega_b, h=h, sigma8=0.8, n_s=0.96,
                        background={'a': a,
                                'chi': distance,
                                'h_over_h0': E_of_z},
                        pk_linear={'a': a,
                        'k': k,
                        'delta_matter:delta_matter': Pk_lin}, 
                        pk_nonlin={'a': a,
                        'k': k,
                        'delta_matter:delta_matter': Pk_nonlin}
                    )
                
                else:
                    cosmo = ccl.CosmologyCalculator(
                        Omega_c=Omega_c, Omega_b=Omega_b, h=h, sigma8=0.8, n_s=0.96,
                        background={'a': a,
                                'chi': distance,
                                'h_over_h0': E_of_z},
                        pk_linear={'a': a,
                        'k': k,
                        'delta_matter:delta_matter': Pk_lin}
                    )                    

        state['CCL'] = {'cosmo': cosmo}
        for required_result, method in self._required_results.items():
            state['CCL'][required_result] = method(cosmo)

    def get_CCL(self):
        """
        Get dictionary of CCL computed quantities.
        results['cosmo'] contains the initialized CCL Cosmology object.
        Other entries are computed by methods passed in as the requirements

        :return: dict of results
        """
        return self._current_state['CCL']

class CrossCorrelationLikelihood(GaussianLikelihood):
    # Cross and auto data
    # Requires some 

    auto_file: str = 'soliket/data/xcorr_simulated/clgg_noiseless.txt'
    cross_file: str = 'soliket/data/xcorr_simulated/clkg_noiseless.txt'
    dndz_file: str = 'soliket/data/xcorr_simulated/dndz.txt'

    params = {'b1': 1, 's1': 0.4} 

    def initialize(self):
        self.dndz = np.loadtxt(self.dndz_file)

        x,y,dy = self._get_data()
        cov = np.diag(dy**2)
        self.data = GaussianData("CrossCorrelation", x, y, cov)

    def get_requirements(self):
        return {'CCL': {"kmax": 10,
                        "nonlinear": True}}

    def _get_data(self, **params_values):
        data_auto = np.loadtxt(self.auto_file)
        data_cross = np.loadtxt(self.cross_file)

        # Get data
        self.ell_auto = data_auto[0]
        cl_auto = data_auto[1]
        cl_auto_err = data_auto[2]

        self.ell_cross = data_cross[0]
        cl_cross = data_cross[1]
        cl_cross_err = data_cross[2]  

        x = np.concatenate([self.ell_auto, self.ell_cross])
        y = np.concatenate([cl_auto, cl_cross])
        dy = np.concatenate([cl_auto_err, cl_cross_err])

        return x, y, dy

    def _get_theory(self, **params_values):
        cosmo = self.provider.get_CCL()['cosmo']

        tracer_g = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz = self.dndz.T, 
                                            bias =(self.dndz[:,0], params_values['b1']*np.ones(len(self.dndz[:,0]))), 
                                            mag_bias = (self.dndz[:,0], params_values['s1']*np.ones(len(self.dndz[:,0])))
                                            )
        tracer_k = ccl.CMBLensingTracer(cosmo, z_source = 1060)

        cl_gg = ccl.cls.angular_cl(cosmo, tracer_g, tracer_g, self.ell_auto) #+ 1e-7
        cl_kg = ccl.cls.angular_cl(cosmo, tracer_k, tracer_g, self.ell_cross)    

        return np.concatenate([cl_gg, cl_kg])

    def logp(self, **params_values):
        theory = self._get_theory(**params_values)
        return self.data.loglike(theory)