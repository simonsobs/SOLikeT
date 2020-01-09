import unittest
import numpy as np

from cobaya.yaml import yaml_load
from cobaya.model import get_model


def get_demo_lensing_model():
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
      n_s: 0.965
    """

    info = yaml_load(info_yaml)
    model = get_model(info)
    return model


class LikeTest(unittest.TestCase):
    def test_cobaya(self):
        model = get_demo_lensing_model()
        lnl = model.loglike()[0]

        assert np.isfinite(lnl)
