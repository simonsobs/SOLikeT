import pkgutil
import pytest

from cobaya.yaml import yaml_load
from cobaya.run import run


@pytest.mark.parametrize("lhood", ["mflike", "lensing", "lensing_lite", "multi", "cross_correlation"])
def test_evaluate(lhood):
    info = yaml_load(pkgutil.get_data("soliket", f"tests/test_{lhood}.yaml"))
    info["force"] = True
    info['sampler'] = {'evaluate': {}}

    updated_info, sampler = run(info)

@pytest.mark.parametrize("lhood", ["mflike", "lensing", "lensing_lite", "multi", "cross_correlation"])
def test_mcmc(lhood):
    info = yaml_load(pkgutil.get_data("soliket", f"tests/test_{lhood}.yaml"))
    info["force"] = True
    info['sampler'] = {'mcmc': {'max_samples': 10, 'max_tries': 1000}}

    updated_info, sampler = run(info)