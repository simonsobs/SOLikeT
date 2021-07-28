import numpy as np
from soliket.ccl import CCL
from soliket.cross_correlation import CrossCorrelationLikelihood
from cobaya.model import get_model
from cobaya.likelihood import Likelihood

def test_cross_correlation():
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
                    "tau": 0,
                    "b1": 1,
                    "s1": 0.4},
            "likelihood": {"CrossCorrelationLikelihood": CrossCorrelationLikelihood},
            "theory": {
                "camb": None,
                "ccl": {"external": CCL, "nonlinear": False}
            },
            "debug": False, "stop_at_error": True}

    model = get_model(info)
    loglikes, derived = model.loglikes()
    assert np.isclose(loglikes[0], 88.2, atol = .2, rtol = 0.)

