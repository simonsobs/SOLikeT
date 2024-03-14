import pkgutil
import pytest
from cobaya.yaml import yaml_load
from cobaya.run import run
from cobaya.tools import resolve_packages_path

packages_path = resolve_packages_path()


@pytest.mark.parametrize("lhood",
                         ["mflike",
                          "lensing",
                          "lensing_lite",
                          "multi",
                          # "galaxykappa",
                          # "shearkappa"
                          # "xcorr"
                          ])
def test_evaluate(lhood):
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
                          # "galaxykappa",
                          # "shearkappa"
                          # "xcorr"
                          ])
def test_mcmc(lhood):
    info = yaml_load(pkgutil.get_data("soliket", f"tests/test_{lhood}.yaml"))
    info["force"] = True
    info['sampler'] = {'mcmc': {'max_samples': 10, 'max_tries': 1000}}

    from cobaya.install import install
    install(info, path=packages_path, skip_global=True)

    updated_info, sampler = run(info)
