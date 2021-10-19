# pytest -k bias -v .

import pytest
import numpy as np

from cobaya.model import get_model

def test_bias_import():
    from soliket.bias import Bias

def test_linear_bias_import():
    from soliket.bias import Linear_bias

def test_linear_bias():

    from soliket.bias import Linear_bias

    info = {"params": {"b_lin": 1.,
                       "ombh2": 0.0245,
                       "omch2": 0.1225,
                       "tau": 0.05},
            "likelihood": {"one" : None},
            "theory": {#"camb" : None,
                       "linear_bias": {"external": Linear_bias}
                       },
            "sampler": {"evaluate": None}
           }

    model = get_model(info)