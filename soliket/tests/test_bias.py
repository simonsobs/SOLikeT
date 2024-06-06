import numpy as np
from cobaya.model import get_model

bias_params = {
    "b_lin": 1.1
}


def test_bias_import():
    from soliket.bias import Bias  # noqa F401


def test_linear_bias_import():
    from soliket.bias import Linear_bias  # noqa F401


def test_linear_bias_model(evaluate_one_info, test_cosmology_params):

    from soliket.bias import Linear_bias

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["params"].update(bias_params)
    evaluate_one_info["theory"] = {
                   "camb": None,
                   "linear_bias": {"external": Linear_bias}
                   }

    model = get_model(evaluate_one_info)  # noqa F841


def test_linear_bias_compute_grid(evaluate_one_info, test_cosmology_params):

    from soliket.bias import Linear_bias

    evaluate_one_info["params"] = test_cosmology_params
    evaluate_one_info["params"].update(bias_params)
    evaluate_one_info["theory"] = {
                   "camb": None,
                   "linear_bias": {"external": Linear_bias}
                   }

    model = get_model(evaluate_one_info)
    model.add_requirements({"Pk_grid": {"z": 0., "k_max": 10.,
                                        "nonlinear": False,
                                        "vars_pairs": ('delta_tot', 'delta_tot')
                                        },
                             "Pk_gg_grid": None,
                             "Pk_gm_grid": None
                            })

    model.logposterior(evaluate_one_info['params'])  # force computation of model

    lhood = model.likelihood['one']

    k, z, Pk_mm_lin = lhood.provider.get_Pk_grid(var_pair=('delta_tot', 'delta_tot'),
                                                 nonlinear=False)

    Pk_gg = lhood.provider.get_Pk_gg_grid()
    Pk_gm = lhood.provider.get_Pk_gm_grid()

    assert np.allclose(Pk_mm_lin * evaluate_one_info["params"]["b_lin"]**2., Pk_gg)
    assert np.allclose(Pk_mm_lin * evaluate_one_info["params"]["b_lin"], Pk_gm)
