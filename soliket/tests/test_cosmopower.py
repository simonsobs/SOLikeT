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

des_params = {
              'DES_DzL1': 0.001,
              'DES_DzL2': 0.002,
              'DES_DzL3': 0.001,
              'DES_DzL4': 0.003,
              'DES_DzL5': 0,
              'DES_b1': 1.45,
              'DES_b2': 1.55,
              'DES_b3': 1.65,
              'DES_b4': 1.8,
              'DES_b5': 2.0,
              'DES_DzS1': -0.001,
              'DES_DzS2': -0.019,
              'DES_DzS3': 0.009,
              'DES_DzS4': -0.018,
              'DES_m1': 0.012,
              'DES_m2': 0.012,
              'DES_m3': 0.012,
              'DES_m4': 0.012,
              'DES_AIA': 1,
              'DES_alphaIA': 1,
              'DES_z0IA': 0.62,
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

def lss_part_likelihood(_self):
    results = _self.provider.get_Pk_interpolator().P(0.1, 1.0)
    # results = _self.provider.get_Pk_grid()
    return 1


info_dict_pk = {
        "params": fiducial_params,
        "likelihood":  {'lss': {'external': lss_part_likelihood, 'requires': {'Pk_interpolator': {'z' : np.linspace(0.0, 2, 128),
                                                                              'k_max': 1.}}}
        },
        "theory": {
        "soliket.CosmoPower": {
            "soliket_data_path": "soliket/data/CosmoPower",
            "stop_at_error": True,
            "provides": 'Pk_grid',
            }
        # 'camb': {"stop_at_error": True,}
        }
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


@pytest.mark.skipif(not HAS_COSMOPOWER, reason='test requires cosmopower')
def test_cosmopower_pkgrid():

    model_cp = get_model(info_dict_pk)

    logL_cp = float(model_cp.loglikes({})[0])

    assert np.isfinite(logL_cp)

