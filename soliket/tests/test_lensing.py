import os
import tempfile
import pytest
import numpy as np
from cobaya.yaml import yaml_load
from cobaya.model import get_model

try:
    import classy  # noqa F401
except ImportError:
    boltzmann_codes = ["camb"]
else:
    boltzmann_codes = ["camb", "classy"]

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "lensing_packages"
)

fiducial_params = {
    "ombh2": 0.02219218,
    "omch2": 0.1203058,
    "H0": 67.02393,
    "tau": 0.6574325e-01,
    "nnu": 3.046,
    "num_massive_neutrinos": 1,
    "As": 2.15086031154146e-9,
    "ns": 0.9625356e00,
}

info = {"theory": {"camb": {"extra_args": {"kmax": 0.9}}}}
info['params'] = fiducial_params


def test_lensing_import(request):
    from soliket.lensing import lensing


def test_lensing_like(request):
    from soliket.lensing import lensing

    info["likelihood"] = {
        "LensLikelihood": {"external": lensing.LensingLikelihood},
        "soliket.utils.OneWithCls": {"lmax": 10000}}

    from cobaya.install import install
    install(info, path=packages_path, skip_global=True)

    model = get_model(info)
    loglikes, derived = model.loglikes()
    assert np.isclose(loglikes[0], 335.85600978, atol=0.2, rtol=0.0)
