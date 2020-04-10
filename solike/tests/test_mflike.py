"""
Make sure that this returns the same result as original mflike.MFLike from LAT_MFlike repo
"""
import os
import pytest
import numpy as np

from cobaya.model import get_model


cosmo_params = {
    "cosmomc_theta": 0.0104085,
    "As": 2.0989031673191437e-09,
    "ombh2": 0.02237,
    "omch2": 0.1200,
    "ns": 0.9649,
    "Alens": 1.0,
    "tau": 0.0544,
}

nuisance_params = {
    "a_tSZ": 3.30,
    "a_kSZ": 1.60,
    "a_p": 6.90,
    "beta_p": 2.08,
    "a_c": 4.90,
    "beta_c": 2.20,
    "n_CIBC": 1.20,
    "a_s": 3.10,
    "T_d": 9.60,
}


def get_demo_mflike_model(orig=False):

    # Choose implementation
    if orig:
        lhood = "mflike.MFLike"
    else:
        lhood = "solike.mflike.MFLike"

    mflike_config = {lhood: {"sim_id": 0, "select": "tt-te-ee", "stop_at_error": True}}

    info = {
        "params": {**cosmo_params, **nuisance_params},
        "likelihood": mflike_config,
        "theory": {"camb": None},
        "modules": os.getenv("COBAYA_MODULES", "/Users/tmorton/cosmology/modules"),
    }
    model = get_model(info)

    return model

@pytest.mark.skip(reason="still in development")
def test_mflike():
    model_local = get_demo_mflike_model()
    model_orig = get_demo_mflike_model(orig=True)

    loglike_local = model_local.loglikes({}, cached=False)[0]  # should be -1384.34401843
    loglike_orig = model_orig.loglikes({}, cached=False)[0]

    assert np.isclose(loglike_local, loglike_orig)
