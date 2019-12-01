import unittest
import numpy as np

class LikeTest(unittest.TestCase):

    def test_cobaya(self):
        from cobaya.yaml import yaml_load
        from cobaya.model import get_model

        info_yaml = r"""
        likelihood:
            solike.SimulatedLensingLiteLikelihood:
                sim_number: 1

        theory:
            classy:
                extra_args:
                    output: lCl, tCl

        params:
          n_s: 0.965
        """

        info = yaml_load(info_yaml)
        model = get_model(info)
        assert np.isfinite(model.loglike()[0])
