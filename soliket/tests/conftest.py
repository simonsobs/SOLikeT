import pytest


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
