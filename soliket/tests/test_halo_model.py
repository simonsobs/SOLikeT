import copy
import numpy as np
from cobaya.model import get_model
import pytest


def test_halomodel_import():
    from soliket.halo_model import HaloModel  # noqa F401


def test_pyhalomodel_import():
    from soliket.halo_model import HaloModel_pyhm  # noqa F401


def test_wrong_types():
    from soliket.halo_model import HaloModel, HaloModel_pyhm

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

    base_case_halo_model_pyhm = {
        "hmf_name": "some_name",
        "hmf_Dv": 1.0,
        "Mmin": 1.0,
        "Mmax": 1.0,
        "nM": 10
    }
    wrong_type_cases_halo_model_pyhm = {
        "hmf_name": 123,
        "hmf_Dv": "not_a_float",
        "Mmin": "not_a_float",
        "Mmax": "not_a_float",
        "nM": "not_an_int"
    }

    for key, wrong_value in wrong_type_cases_halo_model_pyhm.items():
        case = copy.deepcopy(base_case_halo_model_pyhm)
        case[key] = wrong_value
        with pytest.raises(TypeError):
            _ = HaloModel_pyhm(**case)


def test_pyhalomodel_model(evaluate_one_info, test_cosmology_params):
    from soliket.halo_model import HaloModel_pyhm

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["theory"] = {
        "camb": {"stop_at_error": True},
        "halo_model": {"external": HaloModel_pyhm, "stop_at_error": True},
    }

    model = get_model(evaluate_one_info)  # noqa F841


def test_pyhalomodel_compute_mm_grid(evaluate_one_info, test_cosmology_params):
    from soliket.halo_model import HaloModel_pyhm

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["theory"] = {
        "camb": None,
        "halo_model": {"external": HaloModel_pyhm},
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
    # this number derives from the Pk[m-m]
    # calculated in demo-basic.ipynb of the pyhalomodel repo
    assert np.isclose(Pk_mm_hm[0, k > 1.0e-4][0], 3836.7570936793963, rtol=1.0e-3)
