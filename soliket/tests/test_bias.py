# pytest -k bias -v .

import pdb
import pytest
import numpy as np

from cobaya.model import get_model
from cobaya.run import run

info = {"params": {
                   "b_lin": 1.0,
                    "b1g1": 1.1,
                    "b2g1": 1.0,
                    "b1g2": 1.0,
                    "b2g2": 0.0,
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


def test_LPT_bias_compute_grid():

    skip_lpt = pytest.importorskip("velocileptors") # noqa F841
    from soliket.bias import LPT_bias

    info["theory"] = {
               "camb": None,
               "LPT_bias": {"external": LPT_bias,
                            "nonlinear": True}
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

    k, z, Pk_mm_lin = lhood.provider.get_Pk_grid(var_pair=('delta_tot', 'delta_tot'),
                                                 nonlinear=False)
    k_nl, z_nl, Pk_mm_nl = lhood.provider.get_Pk_grid(var_pair=('delta_tot', 'delta_tot'),
                                                      nonlinear=True)

    from matplotlib import pyplot as plt
    plt.ion()
    plt.loglog(k, Pk_mm_lin[0], label='$P_{lin}(k)$')
    plt.loglog(k_nl, Pk_mm_nl[0], '--', label='$P_{HaloFit}(k)$')
    plt.loglog(k_nl, Pk_gm[0], label='$P_{gm}(k)$')
    plt.loglog(k_nl, Pk_gg[0], label='$P_{gg}(k)$')
    plt.xlabel('$k\,$[Mpc$^{-1}$]')
    plt.ylabel('$P(k)$')
    plt.title('velocileptors power spectra')
    plt.legend()
    plt.xlim([1.e-3, 1.e0])
    plt.ylim([1.e3, 1.e5])
    plt.savefig('plots/lpt-bias-pk.png', dpi=300, bbox_inches='tight')

    # assert np.isclose(Pk_gg.sum(), 493326018.33041435)
    # assert np.isclose(Pk_gm.sum(), 429303138.83088464)
