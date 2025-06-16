import importlib

import numpy as np
from cobaya.model import get_model


def test_halomodel_import(check_skip_pyhalomodel):
    _ = importlib.import_module("soliket.halo_model").HaloModel


def test_pyhalomodel_import(check_skip_pyhalomodel):
    _ = importlib.import_module("soliket.halo_model").HaloModel_pyhm


def test_pyhalomodel_model(
    evaluate_one_info, test_cosmology_params, check_skip_pyhalomodel
):
    from soliket.halo_model import HaloModel_pyhm

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["theory"] = {
        "camb": {"stop_at_error": True},
        "halo_model": {"external": HaloModel_pyhm, "stop_at_error": True},
    }

    _ = get_model(evaluate_one_info)


def test_pyhalomodel_compute_mm_grid(
    evaluate_one_info, test_cosmology_params, check_skip_pyhalomodel
):
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
