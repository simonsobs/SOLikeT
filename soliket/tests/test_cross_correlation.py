import numpy as np
import os
import pytest
from soliket.ccl import CCL
from cobaya.model import get_model

auto_file = 'soliket/data/xcorr_simulated/clgg_noiseless.txt'
cross_file = 'soliket/data/xcorr_simulated/clkg_noiseless.txt'
dndz_file = 'soliket/data/xcorr_simulated/dndz.txt'
sacc_file = 'soliket/tests/data/des_s-act_kappa.toy-sim.sacc.fits'

cosmo_params = {"Omega_c": 0.25, "Omega_b": 0.05, "h": 0.67, "n_s": 0.96}

info = {
    "params": {
        "omch2": cosmo_params["Omega_c"] * cosmo_params["h"] ** 2.0,
        "ombh2": cosmo_params["Omega_b"] * cosmo_params["h"] ** 2.0,
        "H0": cosmo_params["h"] * 100,
        "ns": cosmo_params["n_s"],
        "As": 2.2e-9,
        "tau": 0,
        "b1": 1,
        "s1": 0.4,
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

    info["likelihood"] = {
        "GalaxyKappaLikelihood": {"external": GalaxyKappaLikelihood,
                                  "datapath": None,
                                  'cross_file': os.path.join(request.config.rootdir,
                                                             cross_file),
                                  'auto_file': os.path.join(request.config.rootdir,
                                                            auto_file),
                                  'dndz_file': os.path.join(request.config.rootdir,
                                                            dndz_file)}
    }

    model = get_model(info) # noqa F841


# @pytest.mark.xfail(reason="data file not in repo")
def test_shearkappa_model(request):

    from soliket.cross_correlation import ShearKappaLikelihood

    info["likelihood"] = {"ShearKappaLikelihood":
                          {"external": ShearKappaLikelihood,
                           "datapath": os.path.join(request.config.rootdir, sacc_file)}}

    model = get_model(info) # noqa F841


def test_galaxykappa_like(request):

    from soliket.cross_correlation import GalaxyKappaLikelihood

    info["likelihood"] = {
        "GalaxyKappaLikelihood": {"external": GalaxyKappaLikelihood,
                                  "datapath": None,
                                  'cross_file': os.path.join(request.config.rootdir,
                                                             cross_file),
                                  'auto_file': os.path.join(request.config.rootdir,
                                                            auto_file),
                                  'dndz_file': os.path.join(request.config.rootdir,
                                                            dndz_file)}
    }

    model = get_model(info)
    loglikes, derived = model.loglikes()
    assert np.isclose(loglikes[0], 88.2, atol=0.2, rtol=0.0)


# @pytest.mark.xfail(reason="data file not in repo")
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
                      "nnu": 3.046,
                      "s1": 0.4,
                      "b1": 1.0}

    model = get_model(info)
    loglikes, derived = model.loglikes()

    assert np.isclose(loglikes, 637.64473666)

    info["likelihood"]["ShearKappaLikelihood"]["ncovsims"] = 500

    model = get_model(info)
    loglikes, derived = model.loglikes()

    import pdb; pdb.set_trace()


def test_shearkappa_deltaz(request):

    from soliket.cross_correlation import ShearKappaLikelihood

    info["likelihood"] = {"ShearKappaLikelihood":
                          {"external": ShearKappaLikelihood,
                           "datapath": os.path.join(request.config.rootdir, sacc_file),
                           "z_nuisance_mode": "deltaz"}}

    model = get_model(info) # noqa F841
    loglikes, derived = model.loglikes()

    assert np.isfinite(loglikes)


def test_shearkappa_m(request):

    from soliket.cross_correlation import ShearKappaLikelihood

    info["likelihood"] = {"ShearKappaLikelihood":
                          {"external": ShearKappaLikelihood,
                           "datapath": os.path.join(request.config.rootdir, sacc_file),
                           "m_nuisance_mode": True}}

    model = get_model(info) # noqa F841
    loglikes, derived = model.loglikes()

    assert np.isfinite(loglikes)


def test_shearkappa_ia_nla_noevo(request):

    from soliket.cross_correlation import ShearKappaLikelihood

    info["likelihood"] = {"ShearKappaLikelihood":
                          {"external": ShearKappaLikelihood,
                           "datapath": os.path.join(request.config.rootdir, sacc_file),
                           "ia_mode": 'nla-noevo'}}

    model = get_model(info) # noqa F841
    loglikes, derived = model.loglikes()

    assert np.isfinite(loglikes)


def test_shearkappa_ia_nla(request):

    from soliket.cross_correlation import ShearKappaLikelihood

    info["likelihood"] = {"ShearKappaLikelihood":
                          {"external": ShearKappaLikelihood,
                           "datapath": os.path.join(request.config.rootdir, sacc_file),
                           "ia_mode": 'nla'}}

    info["params"]["eta_IA"] = 1.7

    model = get_model(info) # noqa F841
    loglikes, derived = model.loglikes()

    assert np.isfinite(loglikes)


def test_shearkappa_ia_perbin(request):

    from soliket.cross_correlation import ShearKappaLikelihood

    info["likelihood"] = {"ShearKappaLikelihood":
                          {"external": ShearKappaLikelihood,
                           "datapath": os.path.join(request.config.rootdir, sacc_file),
                           "ia_mode": 'nla-perbin'}}

    model = get_model(info) # noqa F841
    loglikes, derived = model.loglikes()

    assert np.isfinite(loglikes)


def test_shearkappa_hmcode(request):

    from soliket.cross_correlation import ShearKappaLikelihood

    info["likelihood"] = {"ShearKappaLikelihood":
                          {"external": ShearKappaLikelihood,
                           "datapath": os.path.join(request.config.rootdir, sacc_file)}}
    info["theory"] = {"camb": {'extra_args': {'halofit_version': 'mead2020_feedback',
                                              'HMCode_logT_AGN': 7.8}},
                      "ccl": {"external": CCL, "nonlinear": False}}

    model = get_model(info) # noqa F841
    loglikes, derived = model.loglikes()

    assert np.isfinite(loglikes)
