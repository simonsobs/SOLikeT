"""
Check that CCL works correctly.
"""
import pytest
import numpy as np
from cobaya.model import get_model
from cobaya.likelihood import Likelihood


class CheckLike(Likelihood):
    """
    This is a mock likelihood that simply forces soliket.CCL to calculate
    a CCL object.
    """
    def logp(self, **params_values):
        ccl = self.theory.get_CCL() # noqa F841
        return -1.0

    def get_requirements(self):
        return {"CCL": None}

fiducial_params = {
    "ombh2": 0.0224,
    "omch2": 0.122,
    "cosmomc_theta": 104e-4,
    "tau": 0.065,
    "ns": 0.9645,
    "logA": 3.07,
    "As": {"value": "lambda logA: 1e-10*np.exp(logA)"}
}

info_dict = {
    "params": fiducial_params,
    "likelihood": {
        "checkLike": {"external": CheckLike}
    },
    "theory": {
        "camb": {
        },
        "soliket.CCL": {
            "kmax": 10.0,
            "nonlinear": True
        }
    }
}


def test_ccl_import(request):
    """
    Test whether we can import pyCCL.
    """
    import pyccl


def test_ccl_cobaya(request):
    """
    Test whether we can call CCL from cobaya.
    """
    model = get_model(info_dict)
    model.loglikes()


def test_ccl_distances(request):
    """
    Test whether the calculated angular diameter distance & luminosity distances
    in CCL have the correct relation.
    """
    model = get_model(info_dict)
    model.loglikes({})
    cosmo = model.provider.get_CCL()["cosmo"]

    z = np.linspace(0.0, 10.0, 100)
    a = 1.0 / (z + 1.0)

    da = cosmo.angular_diameter_distance(a)
    dl = cosmo.luminosity_distance(a)

    assert np.allclose(da * (1.0 + z) ** 2.0, dl)


def test_ccl_pk(request):
    """
    Test whether non-linear Pk > linear Pk in expected regimes.
    """
    model = get_model(info_dict)
    model.loglikes({})
    cosmo = model.provider.get_CCL()["cosmo"]

    k = np.logspace(np.log10(3e-1), 1, 1000)
    pk_lin = cosmo.linear_matter_power(k, a=0.5)
    pk_nonlin = cosmo.nonlin_matter_power(k, a=0.5)

    assert np.all(pk_nonlin > pk_lin)
