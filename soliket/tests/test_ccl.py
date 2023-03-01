"""
Check that CCL works correctly.
"""
import pytest
from cobaya.model import get_model
from cobaya.likelihood import Likelihood

class TestLike(Likelihood):
    def logp(self, **params_values):
        ccl = self.theory.get_CCL()
        return 0.0
    
    def get_requirements(self):
        return { "CCL" : None }

fiducial_params = {
    "ombh2": 0.0224,
    "omch2": 0.122,
    "cosmomc_theta": 104e-4,
    "tau": 0.065,
    "ns": 0.9645,
    "logA": 3.07,
    # derived params
    "As": {"value": "lambda logA: 1e-10 * np.exp(logA)"},
}

info_dict = {
    "params" : fiducial_params,
    "likelihood" : {
        "testLike" : { "external" : TestLike }
    },
    "theory" : {
        "camb" : {
        },
        "soliket.CCL" : {
            "kmax" : 10.0
        }
    }
}

def test_ccl_import(request):
    import pyccl

def test_ccl_cobaya(request):
    """
    Test whether we can call CCL from cobaya.
    """
    model = get_model(info_dict)
    model.loglikes({})
