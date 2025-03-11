import numpy as np
import pytest
from cobaya.model import get_model
from cobaya.yaml import yaml_load

try:
    import classy  # noqa F401
except ImportError:
    boltzmann_codes = ["camb"]
else:
    boltzmann_codes = ["camb", "classy"]


def get_demo_lensing_model(theory):
    if theory == "camb":
        info_yaml = r"""
        likelihood:
            soliket.LensingLiteLikelihood:
                stop_at_error: True

        theory:
            camb:
                extra_args:
                    lens_potential_accuracy: 1

        params:
            ns:
                prior:
                  min: 0.8
                  max: 1.2
            H0:
                prior:
                  min: 40
                  max: 100
        """
    elif theory == "classy":
        info_yaml = r"""
        likelihood:
            soliket.LensingLiteLikelihood:
                stop_at_error: True

        theory:
            classy:
                extra_args:
                    output: lCl, tCl
                path: global

        params:
            n_s:
                prior:
                  min: 0.8
                  max: 1.2
            H0:
                prior:
                  min: 40
                  max: 100

        """

    info = yaml_load(info_yaml)
    model = get_model(info)
    return model


@pytest.mark.parametrize("theory", boltzmann_codes)
def test_lensing(theory):
    model = get_demo_lensing_model(theory)
    ns_param = "ns" if theory == "camb" else "n_s"
    lnl = model.loglike({ns_param: 0.965, "H0": 70})[0]

    assert np.isfinite(lnl)
