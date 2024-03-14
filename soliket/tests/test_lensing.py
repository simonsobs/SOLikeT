import numpy as np
from cobaya.model import get_model

from cobaya.tools import resolve_packages_path

packages_path = resolve_packages_path()
# Cosmological parameters for the test data from SO sims
# See https://github.com/simonsobs/SOLikeT/pull/101 for validation plots
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

    from soliket.lensing import LensingLikelihood # noqa F401


def test_lensing_like(request):

    from cobaya.install import install
    install({"likelihood": {"soliket.lensing.LensingLikelihood": None}},
            path=packages_path, skip_global=False, force=True, debug=True, no_set_global=True)

    from soliket.lensing import LensingLikelihood

    info["likelihood"] = {"LensingLikelihood": {"external": LensingLikelihood}}
    model = get_model(info)
    loglikes, derived = model.loglikes()

    assert np.isclose(loglikes[0], 335.8560097798468, atol=0.2, rtol=0.0)


def test_lensing_ccl_limber(request):
    """
    Test whether the CMB lensing power spectrum predicted by CCL is the same as with CAMB
    """

    from cobaya.install import install
    install({"likelihood": {"soliket.lensing.LensingLikelihood": None}},
            path=packages_path, skip_global=False, force=True, debug=True, no_set_global=True)

    from soliket.lensing import LensingLikelihood
    from copy import deepcopy

    info_dict = deepcopy(info)
    # Neutrino mass put to 0 as far as it is not included in the ccl wrapper
    info_dict['params']["mnu"] = 0
    info_dict['params']["omnuh2"] = 0
    info_dict['likelihood'] = {"LensingLikelihood": {"external": LensingLikelihood}}
    model = get_model(info_dict)
    model.loglikes({})
    cl_camb = model.likelihood['LensingLikelihood']._get_theory()

    info_dict["likelihood"] = {"LensingLikelihood": {"external": LensingLikelihood, 
                                                     "pp_ccl": True}}
    info_dict["theory"]["soliket.CCL"] = {"kmax": 10, "nonlinear": True}
    model = get_model(info_dict)
    model.loglikes({})
    cl_ccl = model.likelihood['LensingLikelihood']._get_theory()

    assert np.any(np.not_equal(cl_ccl, cl_camb))
    assert np.allclose(cl_ccl, cl_camb, rtol=1e-2, atol=0)
