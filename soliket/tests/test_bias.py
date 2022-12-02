# pytest -k bias -v .

import pdb
import pytest
import numpy as np

from cobaya.model import get_model
from cobaya.run import run

info = {"params": {
                   "b_lin": 1.0,
                    "b1g1": 1.0,
                    "b2g1": 1.0,
                    "b1g2": 1.0,
                    "b2g2": 1.0,
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


def test_LPT_bias_model():

    skip_lpt = pytest.importorskip("velocileptors") # noqa F841
    from soliket.bias import LPT_bias

    info["theory"] = {
                   "camb": None,
                   "lpt_bias": {"external": LPT_bias}
                   }

    model = get_model(info)  # noqa F841


@pytest.mark.parametrize("nonlinear_model", [True, False])
def test_LPT_bias_compute_grid(nonlinear_model):

    skip_lpt = pytest.importorskip("velocileptors") # noqa F841
    from soliket.bias import LPT_bias

    info["theory"] = {
               "camb": None,
               "LPT_bias": {"external": LPT_bias,
                            "nonlinear": nonlinear_model}
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

    if nonlinear_model:
        assert np.isclose(Pk_gg.sum(), 489563565.9282355)
        assert np.isclose(Pk_gm.sum(), 425540686.4287062)
    else:
        assert np.isclose(Pk_gg.sum(), 493325841.6713596)
        assert np.isclose(Pk_gm.sum(), 429302962.1718302)


@pytest.mark.parametrize("nonlinear_model", [True, False])
def test_LPT_bias_compute_interpolator(nonlinear_model):

    skip_lpt = pytest.importorskip("velocileptors") # noqa F841
    from soliket.bias import LPT_bias

    info["theory"] = {
               "camb": None,
               "LPT_bias": {"external": LPT_bias,
                            "nonlinear": nonlinear_model}
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

    if nonlinear_model:
        assert np.isclose(Pk_gg, 85288.57867148393)
        assert np.isclose(Pk_gm, 78672.33992060165)
        assert np.isclose(Pk_mm, 78653.14602050644)
    else:
        assert np.isclose(Pk_gg, 85147.06509724)
        assert np.isclose(Pk_gm, 78530.8263451)
        assert np.isclose(Pk_mm, 78794.41444674769)
