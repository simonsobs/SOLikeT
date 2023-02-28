import pkgutil
import pytest
import tempfile
from cobaya.yaml import yaml_load
from cobaya.run import run

import os

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "lensing_packages"
)

@pytest.mark.parametrize("lhood",
                         ["mflike",
                          "lensing",
                          "lensing_lite",
                          "multi",
                          "cross_correlation",
                          # "xcorr"
                          ])
def test_evaluate(lhood):

    # if lhood == "lensing" or lhood == "multi":
    #     pytest.xfail(reason="lensing lhood install failure")

    if lhood == "mflike":
        pytest.skip(reason="don't want to install 300Mb of data!")

    if lhood == "cross_correlation":
        pytest.skip(reason="cannot locate data files")

    info = yaml_load(pkgutil.get_data("soliket", f"tests/test_{lhood}.yaml"))
    info["force"] = True
    info['sampler'] = {'evaluate': {}}

    from cobaya.install import install
    install(info, path=packages_path, skip_global=True)

    updated_info, sampler = run(info)


@pytest.mark.parametrize("lhood",
                         ["mflike",
                          "lensing",
                          "lensing_lite",
                          "multi",
                          "cross_correlation",
                          # "xcorr"
                          ])
def test_mcmc(lhood):

    # if lhood == "lensing" or lhood == "multi":
    #     pytest.xfail(reason="lensing lhood install failure")

    if lhood == "mflike":
        pytest.skip(reason="don't want to install 300Mb of data!")

    if lhood == "cross_correlation":
        pytest.skip(reason="cannot locate data files")

    info = yaml_load(pkgutil.get_data("soliket", f"tests/test_{lhood}.yaml"))
    info["force"] = True
    info['sampler'] = {'mcmc': {'max_samples': 10, 'max_tries': 1000}}

    from cobaya.install import install
    install(info, path=packages_path, skip_global=True)

    updated_info, sampler = run(info)