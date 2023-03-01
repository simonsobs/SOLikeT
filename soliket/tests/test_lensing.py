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
    "ombh2": 0.02219218,
    "omch2": 0.1203058,
    "H0": 67.02393,
    "tau": 0.6574325e-01,
    "nnu": 3.046,
    "mnu": 1.0,
    "As": 2.15086031154146e-9,
    "ns": 0.9625356e00,
}

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

    # code for looking at a validation plot. to be removed.
    # lhood = model.likelihood['LensingLikelihood']

    # x_data, y_data = lhood._get_data()
    # y_th = lhood._get_theory()
    # y_th_owc = lhood_owc._get_theory()

    # x = lhood.data.x
    # y = lhood.data.y
    # cov =lhood.data.cov

    # from matplotlib import pyplot as plt
    # plt.plot(x_data, y_data, 'o')
    # plt.plot(x, y, 'x')
    # plt.plot(x_data, y_th, '-.')
    # plt.legend(['_get_data', '.data', 'computed theory'])
    # plt.xscale('log')
    # plt.xlabel('l')
    # plt.ylabel('C_l')

    # plt.savefig('./lensing_like_testdata.png')

    assert np.isclose(loglikes[0], 44.2959257, atol=0.2, rtol=0.0)
