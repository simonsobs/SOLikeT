import pytest
import sys


def pytest_collection_modifyitems(config, items):
    if sys.platform.startswith('win'):
        skip_on_windows = pytest.mark.skip(reason="Skipped on Windows")
        for item in items:
            if "require_ccl" in item.keywords:
                item.add_marker(skip_on_windows)


@pytest.fixture
def test_cosmology_params():
    params = {}
    params["As"] = 2.15086031154146e-9
    params["ns"] = 0.9625356
    params["ombh2"] = 0.02219218
    params["omch2"] = 0.1203058
    params["H0"] = 67.02393
    params["tau"] = 0.06574325
    params["nnu"] = 3.04
    params["mnu"] = 0.06
    return params


@pytest.fixture
def evaluate_one_info():
    info = {}
    info["likelihood"] = {"one": None}
    info["sampler"] = {"evaluate": None}
    info["debug"] = True
    return info
