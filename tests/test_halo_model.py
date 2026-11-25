import copy
import importlib

import numpy as np
import pytest
from cobaya.model import get_model


def test_halomodel_import():
    _ = importlib.import_module("soliket.halo_model").HaloModel


def test_wrong_types():
    from soliket.halo_model import HaloModel

    base_case_halo_model = {"kmax": 10, "z": 0.5, "extra_args": {}}
    wrong_type_cases_halo_model = {
        "kmax": "not_a_number",
        "z": "not_a_float_or_list_or_array",
        "extra_args": "not_a_dict",
    }

    for key, wrong_value in wrong_type_cases_halo_model.items():
        case = copy.deepcopy(base_case_halo_model)
        case[key] = wrong_value
        with pytest.raises(TypeError):
            _ = HaloModel(**case)


def test_ccl_import(check_skip_pyccl):
    _ = importlib.import_module("soliket.halo_model").HaloModel_ccl


def test_ccl_wrong_types(check_skip_pyccl):
    from soliket.halo_model import HaloModel_ccl

    base_case = {
        "mass_function": "Tinker10",
        "halo_bias": "Tinker10",
        "concentration": "Duffy08",
        "mass_def": "200m",
        "Mmin": 1.0e8,
        "Mmax": 1.0e16,
        "nM": 128,
        "sigma_kmax": 100.0,
        "zmax": 20.0,
    }
    wrong_type_cases = {
        "mass_function": 123,
        "Mmin": "not_a_float",
        "nM": "not_an_int",
        "sigma_kmax": "not_a_float",
        "zmax": "not_a_float",
    }
    for key, wrong_value in wrong_type_cases.items():
        case = copy.deepcopy(base_case)
        case[key] = wrong_value
        with pytest.raises(TypeError):
            _ = HaloModel_ccl(**case)


def test_ccl_compute_mm_grid(
    evaluate_one_info, test_cosmology_params, check_skip_pyccl
):
    from soliket.halo_model import HaloModel_ccl

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["params"]["A_mod"] = 1.0
    evaluate_one_info["theory"] = {
        "camb": None,
        "soliket.CCL": {"nonlinear": False},
        "halo_model": {"external": HaloModel_ccl},
    }

    model = get_model(evaluate_one_info)
    model.add_requirements(
        {
            "Pk_grid": {
                "z": 0.0,
                "k_max": 10.0,
                "nonlinear": False,
                "vars_pairs": ("delta_tot", "delta_tot"),
            },
            "Pk_mm_grid": None,
        }
    )

    model.logposterior(evaluate_one_info["params"])  # force computation of model

    lhood = model.likelihood["one"]

    Pk_mm_hm = lhood.provider.get_Pk_mm_grid()
    k, z, Pk_mm_lin = lhood.provider.get_Pk_grid(
        var_pair=("delta_tot", "delta_tot"), nonlinear=False
    )

    assert np.all(np.isfinite(Pk_mm_hm))
    assert np.all(Pk_mm_hm > 0)
    # regression value for the CCL halo model (pyccl 3.3.4)
    assert np.isclose(Pk_mm_hm[0, k > 1.0e-4][0], 4134.348774213281, rtol=1.0e-3)


def test_ccl_concentration_knob(
    evaluate_one_info, test_cosmology_params, check_skip_pyccl
):
    from soliket.halo_model import HaloModel_ccl

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["params"]["A_mod"] = 1.0
    evaluate_one_info["theory"] = {
        "camb": None,
        "soliket.CCL": {"nonlinear": False},
        "halo_model": {"external": HaloModel_ccl, "concentration": "Bhattacharya13"},
    }

    model = get_model(evaluate_one_info)
    model.add_requirements(
        {
            "Pk_grid": {
                "z": 0.0,
                "k_max": 10.0,
                "nonlinear": False,
                "vars_pairs": ("delta_tot", "delta_tot"),
            },
            "Pk_mm_grid": None,
        }
    )
    model.logposterior(evaluate_one_info["params"])
    Pk_mm_hm = model.likelihood["one"].provider.get_Pk_mm_grid()
    assert np.all(np.isfinite(Pk_mm_hm))
    assert np.all(Pk_mm_hm > 0)


def test_ccl_incompatible_concentration(
    evaluate_one_info, test_cosmology_params, check_skip_pyccl
):
    from cobaya.log import LoggedError

    from soliket.halo_model import HaloModel_ccl

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["params"]["A_mod"] = 1.0
    # Klypin11 is not defined for 200m masses: should fail at initialization with a
    # clear error rather than a cryptic mid-run failure.
    evaluate_one_info["theory"] = {
        "camb": None,
        "soliket.CCL": {"nonlinear": False},
        "halo_model": {
            "external": HaloModel_ccl,
            "concentration": "Klypin11",
            "mass_def": "200m",
        },
    }

    with pytest.raises(LoggedError, match="concentration"):
        get_model(evaluate_one_info)
