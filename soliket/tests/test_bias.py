# pytest -k bias -v .

import pytest
import numpy as np

from cobaya.model import get_model
from cobaya.run import run


def test_bias_import():
    from soliket.bias import Bias


def test_linear_bias_import():
    from soliket.bias import Linear_bias


def test_linear_bias_model():

    from soliket.bias import Linear_bias

    info = {"params": {
                       "b_lin": 1.,
                       "H0": 70.,
                       "ombh2": 0.0245,
                       "omch2": 0.1225,
                       "ns": 0.96,
                       "As": 2.2e-9,
                       "tau": 0.05
                       },
            "likelihood": {"one": None},
            "theory": {"camb": None,
                       "linear_bias": {"external": Linear_bias}
                       },
            "sampler": {"evaluate": None},
            "debug": True
           }

    model = get_model(info)  # noqa F841


def test_linear_bias_run():

    from soliket.bias import Linear_bias

    info = {"params": {
                       "b_lin": 1.,
                       "H0": 70.,
                       "ombh2": 0.0245,
                       "omch2": 0.1225,
                       "ns": 0.96,
                       "As": 2.2e-9,
                       "tau": 0.05
                       },
            "likelihood": {"one": None},
            "theory": {"camb": None,
                       "linear_bias": {"external": Linear_bias}
                       },
            "sampler": {"evaluate": None},
            "debug": True
           }

    model = get_model(info)  # noqa F841
    updated_info, sampler = run(info)
