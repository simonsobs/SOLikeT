import numpy as np
from soliket.ccl import CCL
from cobaya.model import get_model
from cobaya.likelihood import Likelihood
import pyccl as ccl

class Tester(Likelihood):
    params = {'b_hydro': {"prior": {"min": 0, "max": 1}}}

    def get_requirements(self):
        return {'CCL': {"methods": {'test_method': self.test_method},
                        "kmax": 10,
                        "nonlinear": True}}

    def test_method(self, cosmo):
        z_n = np.linspace(0., 1., 200)
        n = np.ones(z_n.shape)
        tracer1 = ccl.WeakLensingTracer(cosmo, dndz=(z_n, n))
        tracer2 = ccl.WeakLensingTracer(cosmo, dndz=(z_n, n))
        ell = np.logspace(np.log10(3), 3)
        cls = ccl.cls.angular_cl(cosmo, tracer1, tracer2, ell)
        return cls

    def logp(self, **pars):
        results = self.provider.get_CCL()
        cls = results['test_method']
        np.testing.assert_almost_equal(cls[0], 1.3478e-08, decimal=8)
        return pars['b_hydro']


cosmo_params = {
    "Omega_c": 0.25,
    "Omega_b": 0.05,
    "h": 0.67,
    "n_s": 0.96
}

info = {"params": {"omch2": cosmo_params['Omega_c'] * cosmo_params['h'] ** 2.,
                   "ombh2": cosmo_params['Omega_b'] * cosmo_params['h'] ** 2.,
                   "H0": cosmo_params['h'] * 100,
                   "ns": cosmo_params['n_s'],
                   "As": 2.2e-9,
                   "tau": 0},
        "likelihood": {"Tester": Tester},
        "theory": {
            "camb": None,
            "ccl": {"external": CCL, "nonlinear": False}
        },
        "debug": False, "stop_at_error": True}

model = get_model(info)
loglikes, derived = model.loglikes({"b_hydro": 0.3})
assert loglikes[0] == 0.3
print('OK')