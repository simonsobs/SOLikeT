import numpy as np
from soliket.ccl import CCL, CrossCorrelationLikelihood
from cobaya.model import get_model
from cobaya.likelihood import Likelihood

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
        "likelihood": {"CrossCorrelationLikelihood": CrossCorrelationLikelihood},
        "theory": {
            "camb": None,
            "ccl": {"external": CCL, "nonlinear": False}
        },
        "debug": False, "stop_at_error": True}

model = get_model(info)
loglikes, derived = model.loglikes()
print('OK')