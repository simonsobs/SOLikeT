import os
import tempfile
import pytest
import numpy as np
from cobaya.yaml import yaml_load
from cobaya.model import get_model

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "lensing_packages"
)

fiducial_params = {
    'omch2': 0.1203058,
    'ombh2': 0.02219218,
    'H0': 67.02393,
    'ns': 0.9625356,
    'As': 2.15086031154146e-9,
    'mnu': 0.06,
    'tau': 0.06574325,
    'nnu': 3.04}

info = {"theory": {"camb": {"extra_args": {"kmax": 0.9}}}}
info['params'] = fiducial_params


def test_lensing_import(request):

    from soliket.lensing import LensingLikelihood


def test_lensing_like(request):

    from cobaya.install import install
    install({"likelihood": {"soliket.lensing.LensingLikelihood": None}},
            path=packages_path, skip_global=False, force=True, debug=True)

    from soliket.lensing import LensingLikelihood

    info["likelihood"] = {"LensingLikelihood": {"external": LensingLikelihood}}
    model = get_model(info)
    loglikes, derived = model.loglikes()

    assert np.isclose(loglikes[0], 335.8560097798468, atol=0.2, rtol=0.0)
