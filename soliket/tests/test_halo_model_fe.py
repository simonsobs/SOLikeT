import numpy as np
from cobaya.model import get_model


def test_hm_fe_import():
    from soliket.halo_model_fe import HaloModel_fe


def test_hm_fe_model(evaluate_one_info, test_cosmology_params):
    from soliket.halo_model_fe import HaloModel_fe

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["theory"] = {
        "camb": {"stop_at_error": True},
        "halo_model_fe": {"external": HaloModel_fe, "stop_at_error": True},
    }
    evaluate_one_info["likelihood"] = {
            "one": {
                "requires": {
                    "Pk_grid": {
                        "z": 0.0,
                        "k_max": 10.0,
                        "nonlinear": False,
                        "var_pairs": ("delta_tot", "delta_tot"),
                    },
                    "Pk_mm_grid": {},
                }
            }
        }


def test_hm_fe_compute_mm_grid(evaluate_one_info, test_cosmology_params):
    from soliket.halo_model_fe import HaloModel_fe

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["theory"] = {
        "camb": None,
        "halo_model_fe": {"external": HaloModel_fe},
    }
    evaluate_one_info["likelihood"] = {
            "one": {
                "requires": {
                    "Pk_grid": {
                        "z": 0.0,
                        "k_max": 10.0,
                        "nonlinear": False,
                        "var_pairs": ("delta_tot", "delta_tot"),
                    },
                    "Pk_mm_grid": {},
                }
            }
        }


    model = get_model(evaluate_one_info)
    model.logposterior(evaluate_one_info["params"])

    lhood = model.likelihood["one"]

    Pk_mm_hm = lhood.provider.get_Pk_mm_grid()
    k, z, Pk_mm_lin = lhood.provider.get_Pk_grid(
        var_pair = ("delta_tot", "delta_tot"), nonlinear=False
    )

    assert np.all(np.isfinite(Pk_mm_hm))
    assert np.isclose(Pk_mm_hm[0, k > 1.0e-4][0], 3118.2290041, rtol=1.0e-3)


                      