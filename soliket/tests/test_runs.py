import pkgutil
import pytest

from cobaya.yaml import yaml_load
from cobaya.run import run


@pytest.mark.parametrize("lhood", ["mflike", "lensing", "lensing_lite", "multi, cross_correlation"])
def test_run(lhood):
    info = yaml_load(pkgutil.get_data("soliket", f"tests/test_{lhood}.yaml"))
    info["force"] = True

    updated_info, sampler = run(info)
