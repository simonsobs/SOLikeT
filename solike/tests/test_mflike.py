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
    "a_tSZ": 3.3044404448917724,
    "a_kSZ": 1.6646620740058649,
    "a_p": 6.912474322461401,
    "beta_p": 2.077474196171309,
    "a_c": 4.88617700670901,
    "beta_c": 2.2030316332596014,
    "n_CIBC": 1.20,
    "a_s": 3.099214100532393,
    "T_d": 9.60,
}

chi2s = {"tt": 1368.5678, "te": 1438.9411, "ee": 1359.1418, "tt-te-et-ee": 2428.0971}
pre = "data_sacc_"


def get_demo_mflike_model(orig=False):

    # Choose implementation
    if orig:
        lhood = "mflike.MFLike"
    else:
        lhood = "solike.mflike.MFLike"

    mflike_config = {
        lhood: {
            "input_file": pre + "00000.fits",
            "cov_Bbl_file": pre + "w_covar_and_Bbl.fits",
            "stop_at_error": True,
        }
    }

    info = {
        "params": {**cosmo_params, **nuisance_params},
        "likelihood": mflike_config,
        "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
        "modules": os.getenv("COBAYA_MODULES", "/Users/tmorton/cosmology/modules"),
    }

    model = get_model(info)

    return model


def test_mflike():
    model_local = get_demo_mflike_model()
    model_orig = get_demo_mflike_model(orig=True)

    loglike_local = model_local.loglikes({}, cached=False)[0].sum()  # [-1]  # should be -1384.34401843
    loglike_orig = model_orig.loglikes({}, cached=False)[0].sum()  # [0]

    assert np.isclose(loglike_local, loglike_orig)
