"""
Check that CosmoPower gives the correct Planck CMB power spectrum.
"""
import os
import pytest
import numpy as np
import matplotlib.pyplot as plt

from cobaya.model import get_model
from soliket.cosmopower import HAS_COSMOPOWER

fiducial_params = {
    "ombh2": 0.0224,
    "omch2": 0.122,
    "h": 0.67,
    "tau": 0.065,
    "ns": 0.9645,
    "logA": 3.07,
    "A_planck": 1.0,
    # derived params
    "As": {"value": "lambda logA: 1e-10 * np.exp(logA)"},
    "H0": {"value": "lambda h: h * 100.0"},
}

info_dict = {
    "params": fiducial_params,
    "likelihood": {
        # This should be installed, otherwise one should install it via cobaya.
        "planck_2018_highl_plik.TTTEEE_lite_native": {"stop_at_error": True}
    },
    "theory": {
        "soliket.CosmoPower": {
            # "soliket_data_path": os.path.normpath(
            #     os.path.join(os.getcwd(), "../data/CosmoPower")
            # ),
            "soliket_data_path": "soliket/data/CosmoPower",
            "stop_at_error": True,
        }
    },
}


@pytest.mark.skipif(not HAS_COSMOPOWER, reason='test requires cosmopower')
def test_cosmopower_theory():
    model_fiducial = get_model(info_dict)   # noqa F841


@pytest.mark.skipif(not HAS_COSMOPOWER, reason='test requires cosmopower')
def test_cosmopower_loglike():
    model_cp = get_model(info_dict)

    logL_cp = float(model_cp.loglikes({})[0])

    assert np.isclose(logL_cp, -295.139)


@pytest.mark.skipif(not HAS_COSMOPOWER, reason='test requires cosmopower')
def test_cosmopower_against_camb():

    info_dict['theory'] = {'camb': {'stop_at_error': True}}
    model_camb = get_model(info_dict)
    logL_camb = float(model_camb.loglikes({})[0])
    camb_cls = model_camb.theory['camb'].get_Cl()

    info_dict['theory'] = {
                           "soliket.CosmoPower": {
                           "soliket_data_path": "soliket/data/CosmoPower",
                           "stop_at_error": True,
                           "extra_args": {'lmax': camb_cls['ell'].max() + 1}}
        }

    model_cp = get_model(info_dict)
    logL_cp = float(model_cp.loglikes({})[0])
    cp_cls = model_cp.theory['soliket.CosmoPower'].get_Cl()

    nanmask = ~np.isnan(cp_cls['tt'])

    assert np.allclose(cp_cls['tt'][nanmask], camb_cls['tt'][nanmask], rtol=1.e-2)
    assert np.isclose(logL_camb, logL_cp, rtol=1.e-1)
