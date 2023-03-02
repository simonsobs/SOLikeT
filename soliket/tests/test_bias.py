# pytest -k bias -v .

import pdb
import pytest
import numpy as np

from cobaya.model import get_model
from cobaya.run import run

info = {"params": {
                    "b_11": 1.0,
                    "b_12": 1.0,
                    "b_21": 1.0,
                    "b_22": 1.0,
                   "H0": 70.,
                   "ombh2": 0.0245,
                   "omch2": 0.1225,
                   "ns": 0.96,
                   "As": 2.2e-9,
                   "tau": 0.05
                   },
        "likelihood": {"one": None},
        "sampler": {"evaluate": None},
        "debug": True
       }


def test_bias_import():
    from soliket.bias import Bias


def test_linear_bias_import():
    from soliket.bias import Linear_bias


def test_linear_bias_model():

    from soliket.bias import Linear_bias

    info["theory"] = {
                   "camb": None,
                   "linear_bias": {"external": Linear_bias}
                   }

    model = get_model(info)  # noqa F841


def test_linear_bias_compute_grid():

    from soliket.bias import Linear_bias

    info["theory"] = {
               "camb": None,
               "linear_bias": {"external": Linear_bias}
               }

    model = get_model(info)  # noqa F841
    model.add_requirements({"Pk_grid": {"z": 0., "k_max": 10.,
                                        "nonlinear": False,
                                        "vars_pairs": ('delta_tot', 'delta_tot')
                                        },
                             "Pk_gg_grid": None,
                             "Pk_gm_grid": None
                            })

    model.logposterior(info['params'])  # force computation of model

    lhood = model.likelihood['one']

    k, z, Pk_mm_lin = lhood.provider.get_Pk_grid(var_pair=('delta_tot', 'delta_tot'),
                                                 nonlinear=False)

    Pk_gg = lhood.provider.get_Pk_gg_grid()
    Pk_gm = lhood.provider.get_Pk_gm_grid()

    assert np.allclose(Pk_mm_lin * info["params"]["b_lin"]**2., Pk_gg)
    assert np.allclose(Pk_mm_lin * info["params"]["b_lin"], Pk_gm)


def test_FastPT_bias_model():

    from soliket.bias import FastPT_bias

    info["theory"] = {
                   "camb": None,
                   "FastPT_bias": {"external": FastPT_bias}
                   }

    model = get_model(info)  # noqa F841


def test_FastPT_bias_compute_grid():

    from soliket.bias import FastPT_bias

    info["theory"] = {
               "camb": None,
               "FastPT_bias": {"external": FastPT_bias,}
               }

    model = get_model(info)  # noqa F841
    model.add_requirements({"Pk_grid": {"z": 0., "k_max": 1.,
                                        "nonlinear": True,
                                        "vars_pairs": ('delta_tot', 'delta_tot'),
                                        },
                             "Pk_gg_grid": None,
                             "Pk_gm_grid": None
                            })

    model.logposterior(info['params'])  # force computation of model

    lhood = model.likelihood['one']

    Pk_gg = lhood.provider.get_Pk_gg_grid()
    Pk_gm = lhood.provider.get_Pk_gm_grid()

    assert np.isclose(Pk_gg.sum(), 368724280.8398774)
    assert np.isclose(Pk_gm.sum(), 377777526.516678)


def test_FastPT_bias_compute_interpolator():

    from soliket.bias import FastPT_bias

    info["theory"] = {
               "camb": None,
               "FastPT_bias": {"external": FastPT_bias}
               }

    model = get_model(info)  # noqa F841
    model.add_requirements({"Pk_grid": {"z": 0., "k_max": 1.,
                                        "nonlinear": True,
                                        "vars_pairs": ('delta_tot', 'delta_tot'),
                                        },
                             "Pk_gg_interpolator": None,
                             "Pk_gm_interpolator": None,
                             "Pk_mm_interpolator": None
                            })

    model.logposterior(info['params'])  # force computation of model

    lhood = model.likelihood['one']

    Pk_gg = lhood.provider.get_Pk_gg_interpolator().P(0.0, 1.e-2)
    Pk_gm = lhood.provider.get_Pk_gm_interpolator().P(0.0, 1.e-2)
    Pk_mm = lhood.provider.get_Pk_mm_interpolator().P(0.0, 1.e-2)

    assert np.isclose(Pk_gg, 78507.04886003392)
    assert np.isclose(Pk_gm, 78531.45751412636)
    assert np.isclose(Pk_mm, 78794.41444674769)