"""
Check that CosmoPower gives the correct Planck CMB power spectrum.
"""
import os
import pytest
import numpy as np
import matplotlib.pyplot as plt

from cobaya.model import get_model

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


def test_cosmopower_theory():
    model_fiducial = get_model(info_dict)


def test_cosmopower_loglike():
    model_cp = get_model(info_dict)

    logL_cp = float(model_cp.loglikes({})[0])

    assert np.isclose(logL_cp, -295.139)


if __name__ == "__main__":
    test_cosmopower_theory()
    test_cosmopower_loglike()
