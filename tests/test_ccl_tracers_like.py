import copy
import importlib
import os

import numpy as np
import pytest
from cobaya.model import get_model

from soliket.ccl import CCL

gammakappa_sacc_file = "tests/data/des_s-act_kappa.toy-sim.sacc.fits"
gkappa_sacc_file = "tests/data/gc_cmass-actdr4_kappa.sacc.fits"

ccl_tracers_params = {
    "b1": 1.0,
    "s1": 0.4,
}
ccl_tracers_theory = {
    "camb": None,
    "ccl": {"external": CCL, "nonlinear": False},
}


def test_galaxykappa_import():
    _ = importlib.import_module("soliket.ccl_tracers").GalaxyKappaLikelihood


def test_shearkappa_import():
    _ = importlib.import_module("soliket.ccl_tracers").ShearKappaLikelihood


def test_galaxykappa_with_wrong_types(request):
    from soliket.ccl_tracers import GalaxyKappaLikelihood

    base_case = {
        "datapath": "valid_path",
        "use_spectra": ["valid"],
        "ncovsims": 5,
        "params": {},
    }
    wrong_type_cases = {
        "datapath": 12345,
        "use_spectra": 12345,
        "ncovsims": "not_an_int",
        "params": "not_a_dict",
    }

    for key, wrong_value in wrong_type_cases.items():
        case = copy.deepcopy(base_case)
        case[key] = wrong_value
        with pytest.raises(TypeError):
            _ = GalaxyKappaLikelihood(**case)


def test_shearkappa_with_wrong_types(request):
    from soliket.ccl_tracers import ShearKappaLikelihood

    base_case = {
        "datapath": "valid_path",
        "use_spectra": ["valid"],
        "ncovsims": 5,
        "params": {},
        "z_nuisance_mode": "valid_str",
        "m_nuisance_mode": True,
        "ia_mode": "valid_str",
    }
    wrong_type_cases = {
        "datapath": 12345,
        "use_spectra": 12345,
        "ncovsims": "not_an_int",
        "params": "not_a_dict",
        "z_nuisance_mode": 12345,
        "m_nuisance_mode": "not_a_bool",
        "ia_mode": 12345,
    }

    for key, wrong_value in wrong_type_cases.items():
        case = copy.deepcopy(base_case)
        case[key] = wrong_value
        with pytest.raises(TypeError):
            _ = ShearKappaLikelihood(**case)


def test_galaxykappa_model(
    request, check_skip_pyccl, evaluate_one_info, test_cosmology_params
):
    from soliket.ccl_tracers import GalaxyKappaLikelihood

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["params"].update(ccl_tracers_params)
    evaluate_one_info["theory"] = ccl_tracers_theory

    evaluate_one_info["likelihood"] = {
        "GalaxyKappaLikelihood": {
            "external": GalaxyKappaLikelihood,
            "datapath": os.path.join(request.config.rootdir, gkappa_sacc_file),
        }
    }

    _ = get_model(evaluate_one_info)


def test_shearkappa_model(
    request, check_skip_pyccl, evaluate_one_info, test_cosmology_params
):
    from soliket.ccl_tracers import ShearKappaLikelihood

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["theory"] = ccl_tracers_theory

    evaluate_one_info["likelihood"] = {
        "ShearKappaLikelihood": {
            "external": ShearKappaLikelihood,
            "datapath": os.path.join(request.config.rootdir, gammakappa_sacc_file),
        }
    }

    _ = get_model(evaluate_one_info)


def test_galaxykappa_like(
    request, check_skip_pyccl, evaluate_one_info, test_cosmology_params, likelihood_refs
):
    from soliket.ccl_tracers import GalaxyKappaLikelihood

    ref = likelihood_refs["galaxykappa"]

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["params"].update(ccl_tracers_params)
    evaluate_one_info["theory"] = ccl_tracers_theory

    evaluate_one_info["likelihood"] = {
        "GalaxyKappaLikelihood": {
            "external": GalaxyKappaLikelihood,
            "datapath": os.path.join(request.config.rootdir, gkappa_sacc_file),
            "use_spectra": [("gc_cmass", "ck_actdr4")],
        }
    }

    model = get_model(evaluate_one_info)
    loglikes, derived = model.loglikes()

    assert np.isclose(loglikes[0], ref["value"], rtol=ref["rtol"], atol=ref["atol"])


def test_shearkappa_like(request, check_skip_pyccl, evaluate_one_info, likelihood_refs):
    from soliket.ccl_tracers import ShearKappaLikelihood

    ref = likelihood_refs["shearkappa"]

    evaluate_one_info["theory"] = ccl_tracers_theory

    rootdir = request.config.rootdir

    cs82_file = "tests/data/cs82_gs-planck_kappa_binned.sim.sacc.fits"
    test_datapath = os.path.join(rootdir, cs82_file)

    evaluate_one_info["likelihood"] = {
        "ShearKappaLikelihood": {
            "external": ShearKappaLikelihood,
            "datapath": test_datapath,
        }
    }

    # Cosmological parameters for the test data, digitized from
    # Fig. 3 and Eq. 8 of Hall & Taylor (2014).
    # See https://github.com/simonsobs/SOLikeT/pull/58 for validation plots
    evaluate_one_info["params"] = {
        "omch2": 0.118,  # Planck + lensing + WP + highL
        "ombh2": 0.0222,
        "H0": 68.0,
        "ns": 0.962,
        "As": 2.1e-9,
        "tau": 0.094,
        "mnu": 0.0,
        "nnu": 3.046,
    }

    model = get_model(evaluate_one_info)
    loglikes, derived = model.loglikes()

    assert np.isclose(loglikes, ref["value"], rtol=ref["rtol"], atol=ref["atol"])


def test_shearkappa_tracerselect(
    request, check_skip_pyccl, evaluate_one_info, test_cosmology_params
):
    import copy

    from soliket.ccl_tracers import ShearKappaLikelihood

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["theory"] = ccl_tracers_theory

    rootdir = request.config.rootdir

    test_datapath = os.path.join(rootdir, gammakappa_sacc_file)

    evaluate_one_info["likelihood"] = {
        "ShearKappaLikelihood": {
            "external": ShearKappaLikelihood,
            "datapath": test_datapath,
            "use_spectra": "all",
        }
    }

    info_onebin = copy.deepcopy(evaluate_one_info)
    info_onebin["likelihood"]["ShearKappaLikelihood"]["use_spectra"] = [
        ("gs_des_bin1", "ck_act")
    ]

    info_twobin = copy.deepcopy(evaluate_one_info)
    info_twobin["likelihood"]["ShearKappaLikelihood"]["use_spectra"] = [
        ("gs_des_bin1", "ck_act"),
        ("gs_des_bin3", "ck_act"),
    ]

    model = get_model(evaluate_one_info)
    loglikes, derived = model.loglikes()

    lhood = model.likelihood["ShearKappaLikelihood"]

    model_onebin = get_model(info_onebin)
    loglikes_onebin, derived_onebin = model_onebin.loglikes()

    lhood_onebin = model_onebin.likelihood["ShearKappaLikelihood"]

    model_twobin = get_model(info_twobin)
    loglikes_twobin, derived_twobin = model_twobin.loglikes()

    lhood_twobin = model_twobin.likelihood["ShearKappaLikelihood"]

    n_ell_perbin = len(lhood.data.x) // 4

    assert n_ell_perbin == len(lhood_onebin.data.x)
    assert np.allclose(lhood.data.y[:n_ell_perbin], lhood_onebin.data.y)

    assert 2 * n_ell_perbin == len(lhood_twobin.data.x)
    assert np.allclose(
        np.concatenate(
            [
                lhood.data.y[:n_ell_perbin],
                lhood.data.y[2 * n_ell_perbin : 3 * n_ell_perbin],
            ]
        ),
        lhood_twobin.data.y,
    )


def test_shearkappa_hartlap(request, check_skip_pyccl, evaluate_one_info):
    from soliket.ccl_tracers import ShearKappaLikelihood

    evaluate_one_info["theory"] = ccl_tracers_theory

    rootdir = request.config.rootdir

    cs82_file = "tests/data/cs82_gs-planck_kappa_binned.sim.sacc.fits"
    test_datapath = os.path.join(rootdir, cs82_file)

    evaluate_one_info["likelihood"] = {
        "ShearKappaLikelihood": {
            "external": ShearKappaLikelihood,
            "datapath": test_datapath,
        }
    }

    # Cosmological parameters for the test data, digitized from
    # Fig. 3 and Eq. 8 of Hall & Taylor (2014).
    # See https://github.com/simonsobs/SOLikeT/pull/58 for validation plots
    evaluate_one_info["params"] = {
        "omch2": 0.118,  # Planck + lensing + WP + highL
        "ombh2": 0.0222,
        "H0": 68.0,
        "ns": 0.962,
        # "As": 2.1e-9,
        "As": 2.5e-9,  # offset the theory to upweight inv_cov in loglike
        "tau": 0.094,
        "mnu": 0.0,
        "nnu": 3.046,
    }

    model = get_model(evaluate_one_info)
    loglikes, derived = model.loglikes()

    evaluate_one_info["likelihood"]["ShearKappaLikelihood"]["ncovsims"] = 5

    model = get_model(evaluate_one_info)
    loglikes_hartlap, derived = model.loglikes()

    assert np.isclose(
        np.abs(loglikes - loglikes_hartlap), 0.0010403, rtol=1.0e-5, atol=1.0e-5
    )


def test_shearkappa_deltaz(
    request, check_skip_pyccl, evaluate_one_info, test_cosmology_params, likelihood_refs
):
    from soliket.ccl_tracers import ShearKappaLikelihood

    ref = likelihood_refs["shearkappa_deltaz"]

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["theory"] = ccl_tracers_theory

    evaluate_one_info["likelihood"] = {
        "ShearKappaLikelihood": {
            "external": ShearKappaLikelihood,
            "datapath": os.path.join(request.config.rootdir, gammakappa_sacc_file),
            "z_nuisance_mode": "deltaz",
        }
    }

    model = get_model(evaluate_one_info)
    loglikes, derived = model.loglikes()

    assert np.isclose(loglikes[0], ref["value"], rtol=ref["rtol"], atol=ref["atol"])


def test_shearkappa_m(
    request, check_skip_pyccl, evaluate_one_info, test_cosmology_params, likelihood_refs
):
    from soliket.ccl_tracers import ShearKappaLikelihood

    ref = likelihood_refs["shearkappa_m"]

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["theory"] = ccl_tracers_theory

    evaluate_one_info["likelihood"] = {
        "ShearKappaLikelihood": {
            "external": ShearKappaLikelihood,
            "datapath": os.path.join(request.config.rootdir, gammakappa_sacc_file),
            "m_nuisance_mode": True,
        }
    }

    model = get_model(evaluate_one_info)
    loglikes, derived = model.loglikes()

    assert np.isclose(loglikes[0], ref["value"], rtol=ref["rtol"], atol=ref["atol"])


def test_shearkappa_ia_nla_noevo(
    request, check_skip_pyccl, evaluate_one_info, test_cosmology_params, likelihood_refs
):
    from soliket.ccl_tracers import ShearKappaLikelihood

    ref = likelihood_refs["shearkappa_ia_nla_noevo"]

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["theory"] = ccl_tracers_theory

    evaluate_one_info["likelihood"] = {
        "ShearKappaLikelihood": {
            "external": ShearKappaLikelihood,
            "datapath": os.path.join(request.config.rootdir, gammakappa_sacc_file),
            "ia_mode": "nla-noevo",
        }
    }

    model = get_model(evaluate_one_info)
    loglikes, derived = model.loglikes()

    assert np.isclose(loglikes[0], ref["value"], rtol=ref["rtol"], atol=ref["atol"])


def test_shearkappa_ia_nla(
    request, check_skip_pyccl, evaluate_one_info, test_cosmology_params, likelihood_refs
):
    from soliket.ccl_tracers import ShearKappaLikelihood

    ref = likelihood_refs["shearkappa_ia_nla"]

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["theory"] = ccl_tracers_theory

    evaluate_one_info["likelihood"] = {
        "ShearKappaLikelihood": {
            "external": ShearKappaLikelihood,
            "datapath": os.path.join(request.config.rootdir, gammakappa_sacc_file),
            "ia_mode": "nla",
        }
    }

    evaluate_one_info["params"]["eta_IA"] = 1.7

    model = get_model(evaluate_one_info)
    loglikes, derived = model.loglikes()

    assert np.isclose(loglikes[0], ref["value"], rtol=ref["rtol"], atol=ref["atol"])


def test_shearkappa_ia_perbin(
    request, check_skip_pyccl, evaluate_one_info, test_cosmology_params, likelihood_refs
):
    from soliket.ccl_tracers import ShearKappaLikelihood

    ref = likelihood_refs["shearkappa_ia_perbin"]

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["theory"] = ccl_tracers_theory

    evaluate_one_info["likelihood"] = {
        "ShearKappaLikelihood": {
            "external": ShearKappaLikelihood,
            "datapath": os.path.join(request.config.rootdir, gammakappa_sacc_file),
            "ia_mode": "nla-perbin",
        }
    }

    model = get_model(evaluate_one_info)
    loglikes, derived = model.loglikes()

    assert np.isclose(loglikes[0], ref["value"], rtol=ref["rtol"], atol=ref["atol"])


def test_shearkappa_hmcode(
    request, check_skip_pyccl, evaluate_one_info, test_cosmology_params, likelihood_refs
):
    from soliket.ccl_tracers import ShearKappaLikelihood

    ref = likelihood_refs["shearkappa_hmcode"]

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["theory"] = ccl_tracers_theory

    evaluate_one_info["likelihood"] = {
        "ShearKappaLikelihood": {
            "external": ShearKappaLikelihood,
            "datapath": os.path.join(request.config.rootdir, gammakappa_sacc_file),
        }
    }
    evaluate_one_info["theory"] = {
        "camb": {
            "extra_args": {"halofit_version": "mead2020_feedback", "HMCode_logT_AGN": 7.8}
        },
        "ccl": {"external": CCL, "nonlinear": False},
    }

    model = get_model(evaluate_one_info)
    loglikes, derived = model.loglikes()

    assert np.isclose(loglikes[0], ref["value"], rtol=ref["rtol"], atol=ref["atol"])


def test_get_ia_bias_variants():
    from soliket.ccl_tracers import ShearKappaLikelihood

    cl = ShearKappaLikelihood.__new__(ShearKappaLikelihood)
    z = np.array([0.1, 0.2, 0.3])
    nz = np.array([1.0, 1.0, 1.0])

    # no IA mode
    cl.ia_mode = None
    assert ShearKappaLikelihood._get_ia_bias(cl, z, nz, "t", {}) is None

    # nla mode
    cl.ia_mode = "nla"
    res = ShearKappaLikelihood._get_ia_bias(cl, z, nz, "t", {"A_IA": 2.0, "eta_IA": 0.5})
    assert isinstance(res, tuple)

    # nla-perbin
    cl.ia_mode = "nla-perbin"
    res2 = ShearKappaLikelihood._get_ia_bias(cl, z, nz, "t", {"t_A_IA": 1.5})
    assert isinstance(res2, tuple)

    # nla-noevo
    cl.ia_mode = "nla-noevo"
    res3 = ShearKappaLikelihood._get_ia_bias(cl, z, nz, "t", {"A_IA": 0.5})
    assert isinstance(res3, tuple)


# --- merged small cross-correlation unit tests (previously in separate files) ---
def _make_fake_sacc_for_merging():
    class Tracer:
        def __init__(self, quantity, z=None, nz=None):
            self.quantity = quantity
            self.z = z
            self.nz = nz

    class FakeSacc:
        def __init__(self):
            self.tracers = {
                "g1": Tracer(
                    "galaxy_shear", z=np.array([0.1, 0.2]), nz=np.array([1.0, 1.0])
                ),
                "k": Tracer("cmb_convergence"),
            }

        def get_tracer_combinations(self):
            return [("g1", "k"), ("k", "g1")]

        def indices(self, tracers=None):
            return [0]

        def get_bandpower_windows(self, idx):
            class BPW:
                values = np.array([10, 20])

                @property
                def weight(self):
                    return np.array([[0.5], [0.5]])

            return BPW()

        def _get_tags_by_index(self, keys, ind):
            return [np.array([10, 20])]

    return FakeSacc()


class _FakeCCL:
    class cells:
        @staticmethod
        def angular_cl(cosmo, t1, t2, ells):
            return np.ones(len(ells)) * 2.0

    @staticmethod
    def CMBLensingTracer(cosmo, z_source=None):
        return "cmb_tracer"

    @staticmethod
    def WeakLensingTracer(cosmo, dndz=None, ia_bias=None):
        return "weak_tracer"


class _FakeProvider:
    def get_param(self, name):
        return 1.0

    def get_CCL(self):
        return {"ccl": _FakeCCL, "cosmo": {}}


def test_shearkappa_galaxy_shear_branch_merged():
    from soliket.ccl_tracers import ShearKappaLikelihood

    lk = ShearKappaLikelihood.__new__(ShearKappaLikelihood)
    lk.sacc_data = _make_fake_sacc_for_merging()
    lk.provider = _FakeProvider()

    lk.z_nuisance_mode = None
    lk.m_nuisance_mode = True
    lk.ia_mode = None

    class SimpleData:
        x = np.array([1.0, 2.0])

    lk.data = SimpleData()

    out = lk._get_theory(**{"g1_m": 0.1})

    assert isinstance(out, np.ndarray)
    assert out.size > 0
    assert np.all(np.isfinite(out))


def test_get_tracer_both_types_merged():
    from soliket.ccl_tracers import ShearKappaLikelihood

    lk = ShearKappaLikelihood.__new__(ShearKappaLikelihood)
    lk.sacc_data = _make_fake_sacc_for_merging()
    lk.provider = _FakeProvider()

    lk.ia_mode = None
    lk.z_nuisance_mode = None
    lk.m_nuisance_mode = None

    ccl, cosmo = lk._get_CCL_results()
    t_cmb = lk._get_tracer(ccl, cosmo, "k", {})
    assert t_cmb is not None
    assert hasattr(t_cmb, "z_source") or isinstance(t_cmb, str)

    t_shear = lk._get_tracer(ccl, cosmo, "g1", {})
    assert t_shear is not None
    assert hasattr(t_shear, "dndz") or isinstance(t_shear, str)
