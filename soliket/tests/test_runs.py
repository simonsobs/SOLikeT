import pkgutil
import pytest
import os

from cobaya.yaml import yaml_load
from cobaya.run import run


@pytest.mark.parametrize("lhood",
                         ["mflike",
                          "lensing",
                          "lensing_lite",
                          "multi",
                          # "cross_correlation",
                          # "xcorr"
                          ])
def test_evaluate(lhood, request):

    if lhood == "lensing" or lhood == "multi":
        pytest.xfail(reason="lensing lhood install failure")

    # info = yaml_load(pkgutil.get_data("soliket", f"tests/test_{lhood}.yaml"))
    info_txt = open(os.path.join(request.config.rootdir,
                                 "soliket/tests/test_{}.yaml".format(lhood)), 'r').read()
    info = yaml_load(info_txt)

    info["force"] = True
    info['sampler'] = {'evaluate': {}}

    updated_info, sampler = run(info)


@pytest.mark.parametrize("lhood",
                         ["mflike",
                          "lensing",
                          "lensing_lite",
                          "multi",
                          # "cross_correlation",
                          # "xcorr"
                          ])
def test_mcmc(lhood, request):

    if lhood == "lensing" or lhood == "multi":
        pytest.xfail(reason="lensing lhood install failure")

    # info = yaml_load(pkgutil.get_data("soliket", f"tests/test_{lhood}.yaml"))
    info_txt = open(os.path.join(request.config.rootdir,
                                 "soliket/tests/test_{}.yaml".format(lhood)), 'r').read()
    info = yaml_load(info_txt)

    info["force"] = True
    info['sampler'] = {'mcmc': {'max_samples': 10, 'max_tries': 1000}}

    updated_info, sampler = run(info)
