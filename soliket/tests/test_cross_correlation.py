import numpy as np
import os
import pytest
from soliket.ccl import CCL
from cobaya.model import get_model

gammakappa_sacc_file = 'soliket/tests/data/des_s-act_kappa.toy-sim.sacc.fits'
gkappa_sacc_file = 'soliket/tests/data/gc_cmass-actdr4_kappa.sacc.fits'

cosmo_params = {"Omega_c": 0.25, "Omega_b": 0.05, "h": 0.67, "n_s": 0.96}

info = {
    "params": {
        "omch2": cosmo_params["Omega_c"] * cosmo_params["h"] ** 2.0,
        "ombh2": cosmo_params["Omega_b"] * cosmo_params["h"] ** 2.0,
        "H0": cosmo_params["h"] * 100,
        "ns": cosmo_params["n_s"],
        "As": 2.2e-9,
        "tau": 0,
        # "b1": 1,
        # "s1": 0.4,
    },
    "theory": {"camb": None, "ccl": {"external": CCL, "nonlinear": False}},
    "debug": False,
    "stop_at_error": True,
}


def test_galaxykappa_import(request):

    from soliket.cross_correlation import GalaxyKappaLikelihood


def test_shearkappa_import(request):

    from soliket.cross_correlation import ShearKappaLikelihood


def test_galaxykappa_model(request):

    from soliket.cross_correlation import GalaxyKappaLikelihood

    info["params"]["b1"] = 2.
    info["params"]["s1"] = 0.4

    info["likelihood"] = {
        "GalaxyKappaLikelihood": {"external": GalaxyKappaLikelihood,
                                  "datapath": os.path.join(request.config.rootdir,
                                                           gkappa_sacc_file)}}

    model = get_model(info) # noqa F841


def test_shearkappa_model(request):

    from soliket.cross_correlation import ShearKappaLikelihood

    # clear out the galaxykappa params if they've been added
    info["params"].pop("b1", None)
    info["params"].pop("s1", None)

    info["likelihood"] = {"ShearKappaLikelihood":
                          {"external": ShearKappaLikelihood,
                           "datapath": os.path.join(request.config.rootdir,
                                                    gammakappa_sacc_file)}}

    model = get_model(info) # noqa F841


def test_galaxykappa_like(request):

    from soliket.cross_correlation import GalaxyKappaLikelihood

    info["params"]["b1"] = 2.
    info["params"]["s1"] = 0.4

    info["likelihood"] = {
        "GalaxyKappaLikelihood": {"external": GalaxyKappaLikelihood,
                                  "datapath": os.path.join(request.config.rootdir,
                                                           gkappa_sacc_file),
                                  "use_spectra": [('gc_cmass', 'ck_actdr4')]}}


    model = get_model(info)
    loglikes, derived = model.loglikes()

    assert np.isclose(loglikes[0], 174.013, atol=0.2, rtol=0.0)


def test_shearkappa_like(request):

    from soliket.cross_correlation import ShearKappaLikelihood

    rootdir = request.config.rootdir

    cs82_file = "soliket/tests/data/cs82_gs-planck_kappa_binned.sim.sacc.fits"
    test_datapath = os.path.join(rootdir, cs82_file)

    info["likelihood"] = {
        "ShearKappaLikelihood": {"external": ShearKappaLikelihood,
                                 "datapath": test_datapath}
    }

    # Cosmological parameters for the test data, digitized from
    # Fig. 3 and Eq. 8 of Hall & Taylor (2014).
    # See https://github.com/simonsobs/SOLikeT/pull/58 for validation plots
    info['params'] = {"omch2": 0.118,  # Planck + lensing + WP + highL
                      "ombh2": 0.0222,
                      "H0": 68.0,
                      "ns": 0.962,
                      "As": 2.1e-9,
                      "tau": 0.094,
                      "mnu": 0.0,
                      "nnu": 3.046}

    model = get_model(info)
    loglikes, derived = model.loglikes()

    assert np.isclose(loglikes, 637.64473666)


def test_shearkappa_tracerselect(request):

    from soliket.cross_correlation import ShearKappaLikelihood
    import copy

    rootdir = request.config.rootdir

    test_datapath = os.path.join(rootdir, gammakappa_sacc_file)

    info["likelihood"] = {
        "ShearKappaLikelihood": {"external": ShearKappaLikelihood,
                                 "datapath": test_datapath,
                                 'use_spectra': 'all'}
    }

    info_onebin = copy.deepcopy(info)
    info_onebin['likelihood']['ShearKappaLikelihood']['use_spectra'] = \
                                                            [('gs_des_bin1', 'ck_act')]
    
    info_twobin = copy.deepcopy(info)
    info_twobin['likelihood']['ShearKappaLikelihood']['use_spectra'] = \
                                                                [
                                                                ('gs_des_bin1', 'ck_act'),
                                                                ('gs_des_bin3', 'ck_act'),
                                                                ]

    model = get_model(info)
    loglikes, derived = model.loglikes()

    lhood = model.likelihood['ShearKappaLikelihood']

    model_onebin = get_model(info_onebin)
    loglikes_onebin, derived_onebin = model_onebin.loglikes()

    lhood_onebin = model_onebin.likelihood['ShearKappaLikelihood']

    model_twobin = get_model(info_twobin)
    loglikes_twobin, derived_twobin = model_twobin.loglikes()

    lhood_twobin = model_twobin.likelihood['ShearKappaLikelihood']

    n_ell_perbin = len(lhood.data.x) // 4

    assert n_ell_perbin == len(lhood_onebin.data.x)
    assert np.allclose(lhood.data.y[:n_ell_perbin], lhood_onebin.data.y)

    assert 2 * n_ell_perbin == len(lhood_twobin.data.x)
    assert np.allclose(np.concatenate([lhood.data.y[:n_ell_perbin],
                                       lhood.data.y[2 * n_ell_perbin:3 * n_ell_perbin]]),
                       lhood_twobin.data.y)


def test_shearkappa_hartlap(request):

    from soliket.cross_correlation import ShearKappaLikelihood

    rootdir = request.config.rootdir

    cs82_file = "soliket/tests/data/cs82_gs-planck_kappa_binned.sim.sacc.fits"
    test_datapath = os.path.join(rootdir, cs82_file)

    info["likelihood"] = {
        "ShearKappaLikelihood": {"external": ShearKappaLikelihood,
                                 "datapath": test_datapath}
    }

    # Cosmological parameters for the test data, digitized from
    # Fig. 3 and Eq. 8 of Hall & Taylor (2014).
    # See https://github.com/simonsobs/SOLikeT/pull/58 for validation plots
    info['params'] = {"omch2": 0.118,  # Planck + lensing + WP + highL
                      "ombh2": 0.0222,
                      "H0": 68.0,
                      "ns": 0.962,
                      # "As": 2.1e-9,
                      "As": 2.5e-9, # offset the theory to upweight inv_cov in loglike
                      "tau": 0.094,
                      "mnu": 0.0,
                      "nnu": 3.046}

    model = get_model(info)
    loglikes, derived = model.loglikes()

    info["likelihood"]["ShearKappaLikelihood"]["ncovsims"] = 5

    model = get_model(info)
    loglikes_hartlap, derived = model.loglikes()

    assert np.isclose(np.abs(loglikes - loglikes_hartlap), 0.0010403,
                      rtol=1.e-5, atol=1.e-5)


def test_shearkappa_deltaz(request):

    from soliket.cross_correlation import ShearKappaLikelihood

    info["likelihood"] = {"ShearKappaLikelihood":
                          {"external": ShearKappaLikelihood,
                           "datapath": os.path.join(request.config.rootdir,
                                                    gammakappa_sacc_file),
                           "z_nuisance_mode": "deltaz"}}

    model = get_model(info) # noqa F841
    loglikes, derived = model.loglikes()

    assert np.isfinite(loglikes)


def test_shearkappa_m(request):

    from soliket.cross_correlation import ShearKappaLikelihood

    info["likelihood"] = {"ShearKappaLikelihood":
                          {"external": ShearKappaLikelihood,
                           "datapath": os.path.join(request.config.rootdir,
                                                    gammakappa_sacc_file),
                           "m_nuisance_mode": True}}

    model = get_model(info) # noqa F841
    loglikes, derived = model.loglikes()

    assert np.isfinite(loglikes)


def test_shearkappa_ia_nla_noevo(request):

    from soliket.cross_correlation import ShearKappaLikelihood

    info["likelihood"] = {"ShearKappaLikelihood":
                          {"external": ShearKappaLikelihood,
                           "datapath": os.path.join(request.config.rootdir,
                                                    gammakappa_sacc_file),
                           "ia_mode": 'nla-noevo'}}

    model = get_model(info) # noqa F841
    loglikes, derived = model.loglikes()

    assert np.isfinite(loglikes)


def test_shearkappa_ia_nla(request):

    from soliket.cross_correlation import ShearKappaLikelihood

    info["likelihood"] = {"ShearKappaLikelihood":
                          {"external": ShearKappaLikelihood,
                           "datapath": os.path.join(request.config.rootdir,
                                                    gammakappa_sacc_file),
                           "ia_mode": 'nla'}}

    info["params"]["eta_IA"] = 1.7

    model = get_model(info) # noqa F841
    loglikes, derived = model.loglikes()

    assert np.isfinite(loglikes)


def test_shearkappa_ia_perbin(request):

    from soliket.cross_correlation import ShearKappaLikelihood

    info["likelihood"] = {"ShearKappaLikelihood":
                          {"external": ShearKappaLikelihood,
                           "datapath": os.path.join(request.config.rootdir,
                                                    gammakappa_sacc_file),
                           "ia_mode": 'nla-perbin'}}

    model = get_model(info) # noqa F841
    loglikes, derived = model.loglikes()

    assert np.isfinite(loglikes)


def test_shearkappa_hmcode(request):

    from soliket.cross_correlation import ShearKappaLikelihood

    info["likelihood"] = {"ShearKappaLikelihood":
                          {"external": ShearKappaLikelihood,
                           "datapath": os.path.join(request.config.rootdir,
                                                    gammakappa_sacc_file)}}
    info["theory"] = {"camb": {'extra_args': {'halofit_version': 'mead2020_feedback',
                                              'HMCode_logT_AGN': 7.8}},
                      "ccl": {"external": CCL, "nonlinear": False}}

    model = get_model(info) # noqa F841
    loglikes, derived = model.loglikes()

    assert np.isfinite(loglikes)
