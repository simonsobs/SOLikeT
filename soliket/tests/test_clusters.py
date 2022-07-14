import numpy as np
import copy
import pytest

from cobaya.model import get_model

fiducial_params = {
    "ombh2": 0.02225,
    "omch2": 0.1198,
    "H0": 67.3,
    "tau": 0.06,
    "As": 2.2e-9,
    "ns": 0.96,
    "mnu": 0.0,
    "nnu": 3.046,
    "omnuh2": 0.,
}

info_unbinned = {
    "params": fiducial_params,
    "likelihood": {"soliket.UnbinnedClusterLikelihood":
    {"stop_at_error": True,
     "theorypred":{"choose_theory":'camb'}}},
    "theory": {
        "camb": {
            "extra_args": {
                "accurate_massive_neutrino_transfers": True,
                "num_massive_neutrinos": 1,
                "redshifts": np.linspace(0, 2, 41),
                "nonlinear": False,
                "kmax": 10.0,
                "dark_energy_model": "ppf",
            }
        },
    },
}

info_binned = copy.copy(info_unbinned)
info_binned['likelihood'] = {"soliket.BinnedClusterLikelihood":
                                    {"stop_at_error": True,
                                     "datapath": './soliket/tests/data/toy_cashc.txt'}}


def test_clusters_unbinned_model():

    model_fiducial = get_model(info_unbinned)


def test_clusters_unbinned_model():

    model_fiducial = get_model(info_unbinned)


def test_clusters_unbinned_loglike():

    model_fiducial = get_model(info_unbinned)

    lnl = model_fiducial.loglikes({})[0]

    print('lnl: ',lnl)
    # exit(0)

    # assert np.isclose(lnl, -885.678)


def test_clusters_unbinned_n_expected():

    model_fiducial = get_model(info_unbinned)

    lnl = model_fiducial.loglikes({})[0]

    like = model_fiducial.likelihood["soliket.UnbinnedClusterLikelihood"]

    print('like._get_n_expected():',like._get_n_expected())
    print('like._get_nz_expected():',like._get_nz_expected())

    assert like._get_n_expected() > 40


def test_clusters_binned_model():

    model_fiducial = get_model(info_binned)

# for debugging purposes:
# test_clusters_unbinned_loglike()
# test_clusters_unbinned_model()
test_clusters_unbinned_n_expected()
