"""
Check that CosmoPower gives the correct Planck CMB power spectrum.
"""

import copy
import importlib
import os

import numpy as np
import pytest
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
            "stop_at_error": True,
            "network_settings": {
                "tt": {
                    "type": "NN",
                    "log": True,
                    "filename": "cmb_TT_NN",
                    # If your network has been trained on (l (l+1) / 2 pi) C_l,
                    # this flag needs to be set.
                    "has_ell_factor": False,
                },
                "ee": {
                    "type": "NN",
                    "log": True,
                    "filename": "cmb_EE_NN",
                    "has_ell_factor": False,
                },
                "te": {
                    "type": "PCAplusNN",
                    # Trained on Cl, not log(Cl)
                    "log": False,
                    "filename": "cmb_TE_PCAplusNN",
                    "has_ell_factor": False,
                },
            },
            "renames": {
                "ombh2": "omega_b",
                "omch2": "omega_cdm",
                "ns": "n_s",
                "logA": "ln10^{10}A_s",
                "tau": "tau_reio",
            },
        }
    },
}


def test_cosmopower_import(check_skip_cosmopower):
    _ = importlib.import_module("soliket.cosmopower").CosmoPower


def test_wrong_types(check_skip_cosmopower):
    from soliket.cosmopower import CosmoPower

    base_case = {
        "network_path": "valid_path",
        "network_settings": {},
        "stop_at_error": True,
        "renames": {},
        "extra_args": {},
    }

    wrong_type_cases = {
        "network_path": 12345,
        "network_settings": "not_a_dict",
        "stop_at_error": "not_a_bool",
        "renames": "not_a_dict",
        "extra_args": "not_a_dict",
    }

    for key, wrong_value in wrong_type_cases.items():
        case = copy.deepcopy(base_case)
        case[key] = wrong_value
        with pytest.raises(TypeError):
            _ = CosmoPower(**case)


def test_cosmopower_theory(request, check_skip_cosmopower, install_planck_lite):
    info_dict["theory"]["soliket.CosmoPower"]["network_path"] = os.path.join(
        request.config.rootdir, "soliket/cosmopower/data/CP_paper"
    )
    _ = get_model(info_dict)


def test_cosmopower_loglike(
    request, check_skip_cosmopower, install_planck_lite, likelihood_refs
):
    ref = likelihood_refs["cosmopower"]
    info_dict["theory"]["soliket.CosmoPower"]["network_path"] = os.path.join(
        request.config.rootdir, "soliket/cosmopower/data/CP_paper"
    )
    model_cp = get_model(info_dict)

    logL_cp = float(model_cp.loglikes({})[0])

    assert np.isclose(logL_cp, ref["value"], rtol=ref["rtol"], atol=ref["atol"])


def test_cosmopower_against_camb(request, check_skip_cosmopower, install_planck_lite):
    info_dict["theory"] = {"camb": {"stop_at_error": True}}
    model_camb = get_model(info_dict)
    logL_camb = float(model_camb.loglikes({})[0])
    camb_cls = model_camb.theory["camb"].get_Cl()

    info_dict["theory"] = {
        "soliket.CosmoPower": {
            "stop_at_error": True,
            "extra_args": {"lmax": camb_cls["ell"].max()},
            "network_path": os.path.join(
                request.config.rootdir, "soliket/cosmopower/data/CP_paper"
            ),
            "network_settings": {
                "tt": {"type": "NN", "log": True, "filename": "cmb_TT_NN"},
                "ee": {"type": "NN", "log": True, "filename": "cmb_EE_NN"},
                "te": {"type": "PCAplusNN", "log": False, "filename": "cmb_TE_PCAplusNN"},
            },
            "renames": {
                "ombh2": "omega_b",
                "omch2": "omega_cdm",
                "ns": "n_s",
                "logA": "ln10^{10}A_s",
                "tau": "tau_reio",
            },
        }
    }

    model_cp = get_model(info_dict)
    logL_cp = float(model_cp.loglikes({})[0])
    cp_cls = model_cp.theory["soliket.CosmoPower"].get_Cl()

    nanmask = ~np.isnan(cp_cls["tt"])

    assert np.allclose(cp_cls["tt"][nanmask], camb_cls["tt"][nanmask], rtol=1.0e-2)
    assert np.isclose(logL_camb, logL_cp, rtol=1.0e-1)


def test_ell_factor_values(check_skip_cosmopower):
    from soliket.cosmopower import CosmoPower

    cp = CosmoPower.__new__(CosmoPower)
    ls = np.array([0, 1, 2, 3], dtype=float)
    tt = CosmoPower.ell_factor(cp, ls, "tt")
    expected_tt = ls * (ls + 1.0) / (2.0 * np.pi)
    assert np.allclose(tt, expected_tt)

    pp = CosmoPower.ell_factor(cp, ls, "pp")
    expected_pp = (ls * (ls + 1.0)) ** 2.0 / (2.0 * np.pi)
    assert np.allclose(pp, expected_pp)


def test_cmb_unit_factor_monkeypatched(check_skip_cosmopower):
    from soliket.cosmopower import CosmoPower

    cp = CosmoPower.__new__(CosmoPower)
    # monkeypatch internal _cmb_unit_factor to avoid dependence on BoltzmannBase
    cp._cmb_unit_factor = lambda units, Tcmb: 10.0

    # 'tt' should multiply the _cmb_unit_factor twice
    res = CosmoPower.cmb_unit_factor(cp, "tt", units="FIRASmuK2", Tcmb=2.7255)
    assert res == 100.0

    # mixed with polarisation 'p' should include 1/sqrt(2pi)
    res2 = CosmoPower.cmb_unit_factor(cp, "tp", units="FIRASmuK2", Tcmb=2.7255)
    # tp => first char t uses _cmb_unit_factor (10), second char p uses 1/sqrt(2pi)
    assert np.isclose(res2, 10.0 * (1.0 / np.sqrt(2.0 * np.pi)))


def test_get_Cl_basic_behavior(check_skip_cosmopower):
    from soliket.cosmopower import CosmoPower

    # create a lightweight dummy object that provides the attributes used by get_Cl
    obj = type("D", (), {})()
    obj.current_state = {
        "ell": np.array([1, 2, 3], dtype=int),
        "tt": np.array([11.0, 22.0, 33.0]),
    }
    obj.extra_args = {"lmax": None}
    obj.networks = {"tt": {"has_ell_factor": False}}
    obj.cmb_unit_factor = lambda k, units, T: 1.0
    obj.log = type("L", (), {"warning": lambda *a, **k: None})()

    cls = CosmoPower.get_Cl(obj, ell_factor=False)
    # cls['ell'] should range from 0..max(ell)
    assert np.array_equal(cls["ell"], np.arange(obj.current_state["ell"].max() + 1))
    # indices 0 and 1 must be zeroed by implementation
    assert cls["tt"][0] == 0.0
    assert cls["tt"][1] == 0.0
    # index 2 should equal the second element of current_state['tt']
    assert cls["tt"][2] == obj.current_state["tt"][1]


def test_get_can_support_parameters_simple(check_skip_cosmopower):
    from soliket.cosmopower import CosmoPower

    cp = CosmoPower.__new__(CosmoPower)
    cp.all_parameters = {"A", "B"}
    out = CosmoPower.get_can_support_parameters(cp)
    assert set(out) == {"A", "B"}


def test_cosmopowerderived_basic_flow(check_skip_cosmopower):
    from soliket.cosmopower import CosmoPowerDerived

    # Build a dummy network that mimics the minimal interface used by CosmoPowerDerived
    class DummyNetwork:
        def __init__(self):
            self.parameters = ["om", "h"]

        def predictions_np(self, params_dict):
            # return a 2D array: shape (1, n_outputs)
            # pretend there are 3 derived parameters
            return np.array([[1.0, 2.0, 3.0]])

        def ten_to_predictions_np(self, params_dict):
            return self.predictions_np(params_dict)

    # Create a minimal CosmoPowerDerived-like object using __new__ and set attributes
    cp = CosmoPowerDerived.__new__(CosmoPowerDerived)
    # wire minimal attributes
    cp.network = DummyNetwork()
    cp.log_data = False
    cp.derived_parameters = ["A", "B", "_"]
    cp.renames = {}
    cp.input_parameters = set(["om", "h"])  # same as network.parameters

    state = {"derived": {}}
    ok = CosmoPowerDerived.calculate(cp, state, want_derived=True, om=0.3, h=0.67)
    assert ok is True
    # The network returned three values, the last derived parameter is '_' so skipped
    assert state["derived"]["A"] == 1.0
    assert state["derived"]["B"] == 2.0

    # get_param should read from current_state['derived'] - use a dummy object
    obj = type("D", (), {})()
    obj.current_state = {"derived": {"om": 0.3}}
    obj.translate_param = lambda p: p
    assert CosmoPowerDerived.get_param(obj, "om") == 0.3

    # get_can_support_parameters should return the input parameters
    out = CosmoPowerDerived.get_can_support_parameters(cp)
    assert set(out) == {"om", "h"}

    # get_requirements should return tuples of (param, None)
    reqs = CosmoPowerDerived.get_requirements(cp)
    assert ("om", None) in reqs and ("h", None) in reqs

    # get_can_provide returns non-empty derived parameters (not '_' or empty)
    provides = CosmoPowerDerived.get_can_provide(cp)
    assert "A" in provides and "B" in provides
