import pkgutil
import pytest

from cobaya.yaml import yaml_load
from cobaya.run import run


@pytest.mark.parametrize("lhood", ["mflike", "lensing", "multi"])
def test_run(lhood):
    info = yaml_load(pkgutil.get_data("solike", f"tests/test_{lhood}.yaml"))
    info["force"] = True

    updated_info, sampler = run(info)
