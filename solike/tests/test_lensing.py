import pytest
import numpy as np

from cobaya.yaml import yaml_load
from cobaya.model import get_model


def get_demo_lensing_model(theory):
    if theory == "camb":
        info_yaml = r"""
        likelihood:
            solike.SimulatedLensingLikelihood:
                sim_number: 1
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
            solike.SimulatedLensingLikelihood:
                sim_number: 1
                stop_at_error: True

        theory:
            classy:
                extra_args:
                    output: lCl, tCl

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


@pytest.mark.parametrize("theory", ["camb", "classy"])
def test_lensing(theory):
    model = get_demo_lensing_model(theory)
    ns_param = "ns" if theory == "camb" else "n_s"
    lnl = model.loglike({ns_param: 0.965, "H0": 70})[0]

    assert np.isfinite(lnl)
