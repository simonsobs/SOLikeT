import numpy as np
import os
import pytest
from soliket.ccl import CCL
from cobaya.model import get_model

gammakappa_sacc_file = 'soliket/tests/data/des_s-act_kappa.toy-sim.sacc.fits'
gkappa_sacc_file = 'soliket/tests/data/gc_cmass-actdr4_kappa.sacc.fits'
galaxykappa_yaml_file = 'soliket/tests/test_galaxykappalike.yaml'
galaxykappa_sacc_file = 'soliket/tests/data/abacus_red-sn+cmbk_cov=sim-noise+theor-err_abacus.fits'

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

def test_galaxykappa_pred(request):
    """
    Test that the prediction from the likelihood is the same as the one from CCL.
    """

    import pyccl as ccl
    import pyccl.nl_pt as pt
    import yaml
    import sacc

    rootdir = request.config.rootdir

    with open(os.path.join(rootdir, galaxykappa_yaml_file), 'r') as f:
        info = yaml.load(f, Loader=yaml.FullLoader)
    info['likelihood']['soliket.cross_correlation.GalaxyKappaLikelihood']['datapath'] = os.path.join(rootdir, galaxykappa_sacc_file)
    print(info)

    params_dict =  {'sigma8': 0.8069016507, 
                    'omch2': 0.1206, 
                    'gcl_cl1_b1': 1.585740073, 
                    'gcl_cl2_b1': 1.813064786, 
                    'gcl_cl3_b1': 2.119892852, 
                    'gcl_cl1_b1p': 1.369028161, 
                    'gcl_cl2_b1p': 1.855865645, 
                    'gcl_cl3_b1p': 2.263006087, 
                    'gcl_cl1_b2': 0.2903571048, 
                    'gcl_cl2_b2': 0.4639643153, 
                    'gcl_cl3_b2': 0.8976356733, 
                    'gcl_cl1_bs': -0.5728063156, 
                    'gcl_cl2_bs': -1.084741786, 
                    'gcl_cl3_bs': -1.709779746, 
                    'gcl_cl1_bk2': 1.557112084, 
                    'gcl_cl2_bk2': 2.356926746, 
                    'gcl_cl3_bk2': 4.006592607, 
                    'mnu': 0.0, 
                    'ombh2': 0.02237,
                    'H0': 67.36,
                    'ns': 0.9649,
                    }

    model = get_model(info) # noqa F841
    loglikes, derived = model.loglikes(params_dict)
    cl_th = model.likelihood['soliket.cross_correlation.GalaxyKappaLikelihood']._get_theory(**params_dict)

    ells_theory, w_bins = model.likelihood['soliket.cross_correlation.GalaxyKappaLikelihood'].get_binning(('cl1', 'cmbk'))

    # Get the theory prediction from CCL
    Omega_c = params_dict['omch2'] / (params_dict['H0']/100.) ** 2.0
    Omega_b = params_dict['ombh2'] / (params_dict['H0']/100.) ** 2.0
    cosmo = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, h=params_dict['H0']/100., sigma8=params_dict['sigma8'], 
                            n_s=params_dict['ns'], m_nu=params_dict['mnu'], matter_power_spectrum='camb', 
                            extra_parameters={"camb": {"halofit_version": "mead2020"}})
    s = sacc.Sacc.load_fits(os.path.join(rootdir, galaxykappa_sacc_file))
    sacc_tr = s.tracers['cl1']
    z_arr = sacc_tr.z
    nz_arr = sacc_tr.nz
    zmean = np.average(z_arr, weights=nz_arr)
    b1z = params_dict['gcl_cl1_b1'] + params_dict['gcl_cl1_b1p']*(z_arr-zmean)
    # Number counts
    ptt_g = pt.PTNumberCountsTracer(b1=(z_arr, b1z), b2=params_dict['gcl_cl1_b2'], 
                                    bs=params_dict['gcl_cl1_bs'], bk2=params_dict['gcl_cl1_bk2'])
    # Matter
    ptt_m = pt.PTMatterTracer()
    # Number counts
    t_g = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z_arr, nz_arr), 
                                bias=(z_arr, np.ones_like(z_arr)), mag_bias=None)
    # Lensing
    t_l = ccl.CMBLensingTracer(cosmo, z_source=1030)
    ptc = pt.LagrangianPTCalculator(log10k_min=-4, log10k_max=2, nk_per_decade=20)
    ptc.update_ingredients(cosmo)
    # Galaxies x matter
    pk_gm = ptc.get_biased_pk2d(ptt_g, tracer2=ptt_m)
    ell = np.arange(5287)
    cls_ccl = ccl.angular_cl(cosmo, t_g, t_l, ells_theory, p_of_k_a=pk_gm)
    cls_ccl_binned = np.dot(w_bins, cls_ccl)

    assert np.allclose(cls_ccl_binned, cl_th[:10], rtol=4.e-4, atol=1.e-8)

