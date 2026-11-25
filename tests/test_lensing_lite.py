import importlib
import pathlib

import numpy as np
import pytest
from cobaya.model import get_model
from cobaya.yaml import yaml_load

from soliket import LensingLiteLikelihood

try:
    _ = importlib.import_module("classy")
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


_ACT_DR6_DATA = str(pathlib.Path(__file__).parent / "data" / "act_dr6_lensing.sacc.fits")

fiducial_params = {
    "omch2": 0.1203058,
    "ombh2": 0.02219218,
    "H0": 67.02393,
    "ns": 0.9625356,
    "As": 2.15086031154146e-9,
    "mnu": 0.06,
    "tau": 0.06574325,
    "nnu": 3.04,
}


def test_lensing_lite_act_dr6(likelihood_refs):
    """Unit test: LensingLiteLikelihood on ACT DR6 data at SO fiducial cosmology."""
    ref = likelihood_refs["lensing_lite_act_dr6"]

    info = {
        "likelihood": {
            "soliket.LensingLiteLikelihood": {
                "stop_at_error": True,
                "datapath": _ACT_DR6_DATA,
                "ncovsims": 792,
            }
        },
        "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
        "params": fiducial_params,
    }

    model = get_model(info)
    loglikes, derived = model.loglikes()

    assert np.isclose(loglikes[0], ref["value"], rtol=ref["rtol"], atol=ref["atol"])


class DummyProviderCl:
    def __init__(self, lmax):
        self.lmax = lmax

    def get_Cl(self, ell_factor=True):
        # return small arrays for pp, tt, ee, te, bb
        size = self.lmax
        return {
            "pp": np.arange(size, dtype=float) + 1.0,
            "tt": (np.arange(size, dtype=float) + 2.0),
            "ee": (np.arange(size, dtype=float) + 3.0),
            "te": (np.arange(size, dtype=float) + 4.0),
            "bb": (np.arange(size, dtype=float) + 5.0),
        }


def test_lensinglite_get_theory_basic():
    lmax = 5
    ll = LensingLiteLikelihood.__new__(LensingLiteLikelihood)
    # set minimal attributes
    ll.provider = DummyProviderCl(lmax)
    ll.lmax = lmax
    ll.ls = np.arange(lmax, dtype=np.longlong)
    ll.use_spectra = ("ck", "ck")
    # make binning matrix identity so binned result equals theory
    ll.binning_matrix = np.eye(lmax)
    # call LensingLiteLikelihood._get_theory which computes binned Clkk from pp
    out = LensingLiteLikelihood._get_theory(ll)
    expected = (ll.ls * (ll.ls + 1)) ** 2 * ll.provider.get_Cl()["pp"][0 : ll.lmax] * 0.25
    # provider returns pp = [1,2,3,4,5], so out should equal that
    assert np.allclose(out, expected)
